# Автоматический деплой Galeries Backend

## Описание

Система автоматического деплоя следит за папкой `/home/nickr/` и при появлении файла `galeries.zip` автоматически:

1. ✅ Останавливает сервер
2. ✅ Создает бэкап текущей версии
3. ✅ Разархивирует новую версию
4. ✅ Запускает установку (SETUP.sh)
5. ✅ Запускает сервер
6. ✅ Удаляет архив
7. ✅ Очищает старые бэкапы (оставляет последние 3)

## Установка

### Вариант 1: Как systemd сервис (рекомендуется)

\`\`\`bash
# На сервере
cd /home/nickr
chmod +x install-auto-deploy.sh
sudo ./install-auto-deploy.sh
\`\`\`

Сервис будет автоматически запускаться при перезагрузке сервера.

### Вариант 2: Ручной запуск

\`\`\`bash
# На сервере
cd /home/nickr
chmod +x auto-deploy.sh
sudo ./auto-deploy.sh
\`\`\`

Скрипт будет работать пока не нажмете Ctrl+C.

## Использование

### С вашего компьютера:

\`\`\`bash
# Создайте архив
cd galeries
zip -r galeries.zip python/ SETUP.sh

# Загрузите на сервер
scp galeries.zip nickr@23.88.61.20:/home/nickr/

# Деплой начнется автоматически!
\`\`\`

### Мониторинг

\`\`\`bash
# Смотреть логи в реальном времени
tail -f /home/nickr/auto-deploy.log

# Статус сервиса
sudo systemctl status galeries-auto-deploy

# Перезапустить сервис
sudo systemctl restart galeries-auto-deploy
\`\`\`

## Управление сервисом

\`\`\`bash
# Остановить автоматический деплой
sudo systemctl stop galeries-auto-deploy

# Запустить
sudo systemctl start galeries-auto-deploy

# Отключить автозапуск
sudo systemctl disable galeries-auto-deploy

# Включить автозапуск
sudo systemctl enable galeries-auto-deploy
\`\`\`

## Бэкапы

Система автоматически создает бэкапы перед каждым деплоем:
- Формат: `python_backup_YYYYMMDD_HHMMSS`
- Хранятся последние 3 бэкапа
- Старые автоматически удаляются

### Восстановление из бэкапа:

\`\`\`bash
cd /home/nickr
sudo systemctl stop galeries-auto-deploy
cd python && ./stop.sh
cd ..
rm -rf python
mv python_backup_20250130_143022 python  # замените на нужный бэкап
cd python && ./start-daemon.sh
sudo systemctl start galeries-auto-deploy
\`\`\`

## Логи

Все действия записываются в `/home/nickr/auto-deploy.log`:

\`\`\`bash
# Последние 50 строк
tail -n 50 /home/nickr/auto-deploy.log

# Следить в реальном времени
tail -f /home/nickr/auto-deploy.log

# Очистить логи
sudo truncate -s 0 /home/nickr/auto-deploy.log
\`\`\`

## Безопасность

⚠️ **Важно**: Скрипт запускается от root для возможности установки системных пакетов.

Убедитесь что:
- Доступ к серверу защищен SSH ключами
- Только доверенные пользователи могут загружать файлы в `/home/nickr/`
- Регулярно проверяйте логи на подозрительную активность

## Устранение проблем

### Сервис не запускается

\`\`\`bash
# Проверить статус
sudo systemctl status galeries-auto-deploy

# Проверить логи systemd
sudo journalctl -u galeries-auto-deploy -n 50

# Проверить логи деплоя
tail -n 100 /home/nickr/auto-deploy.log
\`\`\`

### Деплой не срабатывает

\`\`\`bash
# Проверить что inotify-tools установлен
which inotifywait

# Проверить права на файлы
ls -la /home/nickr/auto-deploy.sh
ls -la /home/nickr/

# Попробовать ручной деплой
cd /home/nickr
sudo ./auto-deploy.sh
\`\`\`

### Ошибка при установке

\`\`\`bash
# Проверить логи установки
cat /home/nickr/auto-deploy.log | grep "SETUP.sh"

# Проверить что SETUP.sh есть в архиве
unzip -l galeries.zip | grep SETUP.sh
\`\`\`

## Тестирование

Для проверки работы системы:

\`\`\`bash
# 1. Создайте тестовый архив
cd /tmp
mkdir -p python
echo "test" > python/test.txt
zip -r galeries.zip python/

# 2. Загрузите на сервер
scp galeries.zip nickr@23.88.61.20:/home/nickr/

# 3. Следите за логами
ssh nickr@23.88.61.20 "tail -f /home/nickr/auto-deploy.log"
\`\`\`

## Версия

Текущая версия: 1.0.0
Совместимость: Galeries Backend v3.2.8+
