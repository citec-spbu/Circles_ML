// Основной класс приложения
class CircleDetectionApp {
    constructor() {
        this.canvas = document.getElementById('resultCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.imageInput = document.getElementById('imageInput');
        this.detectorSelect = document.getElementById('detectorSelect');
        this.parametersList = document.getElementById('parametersList');
        this.resultsInfo = document.getElementById('resultsInfo');
        this.detectorDescription = document.getElementById('detectorDescription');
        this.detectorInfoText = document.getElementById('detectorInfoText');

        this.currentImage = null;
        this.detectionResults = [];
        this.availableDetectors = [];
        this.currentScale = 1;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadAvailableDetectors();
        this.updateParametersUI();
    }

    setupEventListeners() {
        // Обработчик выбора файла
        this.imageInput.addEventListener('change', (e) => {
            this.handleImageUpload(e.target.files[0]);
        });

        // Обработчик выбора детектора
        this.detectorSelect.addEventListener('change', () => {
            this.onDetectorChange();
        });

        // // Кнопка валидации
        // document.getElementById('validateBtn').addEventListener('click', () => {
        //     this.validateConfiguration();
        // });
        //
        // // Кнопка обновления списка детекторов
        // document.getElementById('refreshDetectors').addEventListener('click', () => {
        //     this.refreshDetectors();
        // });
    }

    async refreshDetectors() {
        this.showAlert('Обновление списка детекторов...', 'info');
        await this.loadAvailableDetectors(true);
    }

    async loadAvailableDetectors(forceRefresh = false) {
        try {
            const url = forceRefresh ? '/api/detectors?force_refresh=true' : '/api/detectors';
            const response = await fetch(url);
            const detectors = await response.json();
            this.availableDetectors = detectors;

            this.updateDetectorSelect(detectors);
        } catch (error) {
            console.error('Failed to load detectors:', error);
            this.showAlert('Ошибка загрузки списка детекторов', 'error');
        }
    }

    updateDetectorSelect(detectors) {
        // Обновляем опции выбора
        this.detectorSelect.innerHTML = '<option value="">Выберите детектор...</option>';
        detectors.forEach(detector => {
            const option = document.createElement('option');
            option.value = detector.class_name;
            option.textContent = `${detector.name} (v${detector.version})`;
            option.dataset.modulePath = detector.module_path;
            option.dataset.fullModulePath = detector.full_module_path;
            option.dataset.required = JSON.stringify(detector.required_parameters);
            option.dataset.optional = JSON.stringify(detector.optional_parameters);
            option.dataset.description = detector.description;
            this.detectorSelect.appendChild(option);
        });
    }

    onDetectorChange() {
        const selectedOption = this.detectorSelect.options[this.detectorSelect.selectedIndex];

        if (this.detectorSelect.value === '') {
            this.hideDetectorInfo();
            this.parametersList.innerHTML = '<p>Выберите детектор для настройки параметров</p>';
            return;
        }

        // Показываем информацию о детекторе
        this.showDetectorInfo(selectedOption);

        // Обновляем параметры
        this.updateParametersUI();
    }

    showDetectorInfo(option) {
        const description = option.dataset.description;

        let infoHtml = `<strong>${option.textContent}</strong><br>`;
        infoHtml += `${description}`;

        this.detectorInfoText.innerHTML = infoHtml;
        this.detectorDescription.style.display = 'block';
    }

    hideDetectorInfo() {
        this.detectorDescription.style.display = 'none';
    }

    async validateConfiguration() {
        const detectorConfig = this.getDetectorConfig();
        if (!detectorConfig) return;

        try {
            const formData = new FormData();
            formData.append('detector_config', JSON.stringify(detectorConfig));

            const response = await fetch('/api/validate', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.valid) {
                let message = '✅ Конфигурация корректна';
                if (result.warnings.length > 0) {
                    message += ` (предупреждения: ${result.warnings.join(', ')})`;
                }
                this.showAlert(message, 'success');
            } else {
                this.showAlert(`❌ Ошибки в конфигурации: ${result.errors.join(', ')}`, 'error');
            }

        } catch (error) {
            console.error('Validation error:', error);
            this.showAlert('Ошибка валидации', 'error');
        }
    }

    updateParametersUI() {
        const selectedOption = this.detectorSelect.options[this.detectorSelect.selectedIndex];

        if (this.detectorSelect.value === '') {
            return;
        }

        // Очищаем список параметров
        this.parametersList.innerHTML = '';

        // Создаем параметры из информации о детекторе
        const requiredParams = JSON.parse(selectedOption.dataset.required || '[]');
        const optionalParams = JSON.parse(selectedOption.dataset.optional || '{}');

        // Создаем поля для обязательных параметров
        requiredParams.forEach(param => {
            this.createParameterField(param, '', true);
        });

        // Создаем поля для опциональных параметров
        Object.entries(optionalParams).forEach(([param, defaultValue]) => {
            this.createParameterField(param, defaultValue, false);
        });
    }

    createParameterField(paramName, defaultValue, isRequired) {
        const paramDiv = document.createElement('div');
        paramDiv.className = 'parameter-item';

        const label = document.createElement('label');
        label.textContent = `${paramName} ${isRequired ? '*' : ''}`;
        label.htmlFor = `param_${paramName}`;

        // Определяем тип input на основе значения по умолчанию
        let inputType = 'number';
        let step = 'any';

        if (typeof defaultValue === 'boolean') {
            inputType = 'checkbox';
        } else if (typeof defaultValue === 'string') {
            inputType = 'text';
        } else if (Number.isInteger(defaultValue)) {
            step = '1';
        }

        const input = document.createElement('input');
        input.type = inputType;
        input.id = `param_${paramName}`;
        input.name = paramName;

        if (inputType === 'checkbox') {
            input.checked = defaultValue;
        } else {
            input.value = defaultValue;
            input.step = step;
        }

        input.required = isRequired;

        paramDiv.appendChild(label);
        paramDiv.appendChild(input);
        this.parametersList.appendChild(paramDiv);
    }

    handleImageUpload(file) {
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this.showAlert('Пожалуйста, выберите файл изображения', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.loadImageToCanvas(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    loadImageToCanvas(dataUrl) {
        const img = new Image();
        img.onload = () => {
            // Сохраняем оригинальное изображение для обработки
            this.currentImage = img;

            // Для отображения используем уменьшенную версию (если нужно)
            const maxDisplayWidth = 800;
            const scale = Math.min(maxDisplayWidth / img.width, 1);
            this.currentScale = scale;

            this.canvas.width = img.width * scale;
            this.canvas.height = img.height * scale;

            // Рисуем уменьшенную версию для отображения
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);

            this.detectionResults = [];
            this.clearResults();
        };
        img.src = dataUrl;
    }

    async detectCenters() {
        if (!this.currentImage) {
            this.showAlert('Пожалуйста, загрузите изображение', 'error');
            return;
        }

        if (this.detectorSelect.value === '') {
            this.showAlert('Пожалуйста, выберите детектор', 'error');
            return;
        }

        const detectorConfig = this.getDetectorConfig();
        if (!detectorConfig) return;

        this.showLoading(true);

        try {
            const formData = new FormData();

            // Отправляем оригинальный файл
            const fileInput = document.getElementById('imageInput');
            if (fileInput.files.length === 0) {
                throw new Error('Файл не выбран');
            }
            formData.append('file', fileInput.files[0]);
            formData.append('detector_config', JSON.stringify(detectorConfig));

            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Ошибка детекции');
            }

            const result = await response.json();
            this.handleDetectionResult(result);

        } catch (error) {
            console.error('Detection error:', error);
            this.showAlert(`Ошибка детекции: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    getDetectorConfig() {
        const selectedOption = this.detectorSelect.options[this.detectorSelect.selectedIndex];
        const modulePath = selectedOption.dataset.modulePath;
        const className = selectedOption.value;

        // Собираем параметры
        const parameters = {};
        const paramInputs = this.parametersList.querySelectorAll('input');

        paramInputs.forEach(input => {
            let value;

            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                value = input.value === '' ? '' : parseFloat(input.value);
            } else {
                value = input.value;
            }

            // Добавляем только если значение не пустое (кроме чекбоксов)
            if (input.type !== 'checkbox' && value === '') {
                return;
            }

            parameters[input.name] = value;
        });

        return {
            module_path: modulePath,
            class_name: className,
            parameters: parameters
        };
    }

    handleDetectionResult(result) {
        this.detectionResults = result.centers;

        // Рисуем результаты на canvas с учетом масштаба
        this.drawDetectionResults();

        // Показываем информацию
        this.displayResultsInfo(result);

        // Показываем предупреждения если есть
        if (result.warnings && result.warnings.length > 0) {
            this.showAlert(`Найдено ${result.centers.length} кругов (предупреждения: ${result.warnings.join(', ')})`, 'warning');
        } else {
            this.showAlert(`Найдено ${result.centers.length} кругов`, 'success');
        }
    }

    drawDetectionResults() {
        if (!this.currentImage || !this.detectionResults.length) return;

        // Перерисовываем изображение
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.currentImage, 0, 0, this.canvas.width, this.canvas.height);

        // Рисуем центры и круги с учетом масштаба
        this.detectionResults.forEach((center, index) => {
            const x = center.center_x * this.currentScale;
            const y = center.center_y * this.currentScale;
            const radius = center.radius ? center.radius * this.currentScale : 5;

            // Рисуем круг
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.strokeStyle = '#ff0000';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Рисуем центр
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
            this.ctx.fillStyle = '#00ff00';
            this.ctx.fill();

            // Номер центра
            this.ctx.fillStyle = '#0000ff';
            this.ctx.font = '14px Arial';
            this.ctx.fillText(`${index + 1}`, x + 8, y - 8);

            // Нормаль (если есть и не нулевая)
            if ((center.normal_x !== 0 || center.normal_y !== 0) &&
                !isNaN(center.normal_x) && !isNaN(center.normal_y)) {
                const normalLength = 20;
                this.ctx.beginPath();
                this.ctx.moveTo(x, y);
                this.ctx.lineTo(
                    x + center.normal_x * normalLength,
                    y + center.normal_y * normalLength
                );
                this.ctx.strokeStyle = '#0000ff';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            }
        });
    }

    displayResultsInfo(result) {
        let html = `
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">${result.centers.length}</div>
                    <div class="stat-label">Найдено кругов</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${result.detector_name}</div>
                    <div class="stat-label">Детектор</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">v${result.detector_version}</div>
                    <div class="stat-label">Версия</div>
                </div>
            </div>
        `;

        // Добавляем предупреждения если есть
        if (result.warnings && result.warnings.length > 0) {
            html += `
                <div class="alert alert-warning">
                    <strong>Предупреждения:</strong> ${result.warnings.join(', ')}
                </div>
            `;
        }

        html += `<h4>Результаты детекции:</h4>`;

        if (result.centers.length === 0) {
            html += `<p>Круги не обнаружены.</p>`;
        } else {
            result.centers.forEach((center, index) => {
                html += `
                    <div class="result-item">
                        <strong>Круг ${index + 1}:</strong>
                        <div class="result-coordinates">
                            Центр: (${center.center_x.toFixed(3)}, ${center.center_y.toFixed(3)})
                            ${center.radius ? `Радиус: ${center.radius.toFixed(2)}` : ''}
                            Уверенность: ${(center.confidence * 100).toFixed(1)}%
                        </div>
                        <div>Нормаль: (${center.normal_x.toFixed(3)}, ${center.normal_y.toFixed(3)}, ${center.normal_z.toFixed(3)})</div>
                    </div>
                `;
            });
        }

        this.resultsInfo.innerHTML = html;
    }

    clearResults() {
        this.resultsInfo.innerHTML = '<p>Результатов пока нет. Загрузите изображение и нажмите "Найти центры".</p>';
    }

    showLoading(show) {
        const button = document.querySelector('button[onclick="detectCenters()"]');
        const validateBtn = document.getElementById('validateBtn');

        if (show) {
            button.disabled = true;
            // validateBtn.disabled = true;
            button.textContent = 'Обработка...';
        } else {
            button.disabled = false;
            // validateBtn.disabled = false;
            button.textContent = 'Найти центры';
        }
    }

    showAlert(message, type = 'info') {
        // Удаляем существующие уведомления
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;

        // Вставляем перед контейнером
        document.querySelector('.container').insertBefore(alert, document.querySelector('.upload-config-section'));

        // Автоматически удаляем через 5 секунд
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 5000);
    }
}

// Глобальные функции для HTML
let app;

function initApp() {
    app = new CircleDetectionApp();
}

function detectCenters() {
    if (app) {
        app.detectCenters();
    }
}

function onDetectorChange() {
    if (app) {
        app.onDetectorChange();
    }
}

function validateConfiguration() {
    if (app) {
        app.validateConfiguration();
    }
}

function refreshDetectors() {
    if (app) {
        app.refreshDetectors();
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', initApp);