#!/bin/bash

# Скрипт для создания архива backend для деплоя
# Использование: ./scripts/create-backend-archive.sh

VERSION="v3.2.7"
ARCHIVE_NAME="backend-${VERSION}.zip"

echo "Creating backend archive ${ARCHIVE_NAME}..."

# Создаем временную директорию
TEMP_DIR=$(mktemp -d)
mkdir -p "${TEMP_DIR}/python"

# Копируем все файлы из python/
cp -r python/* "${TEMP_DIR}/python/"

# Создаем архив
cd "${TEMP_DIR}"
zip -r "${ARCHIVE_NAME}" python/

# Перемещаем архив в корень проекта
mv "${ARCHIVE_NAME}" "${OLDPWD}/"

# Удаляем временную директорию
cd "${OLDPWD}"
rm -rf "${TEMP_DIR}"

echo "✓ Archive created: ${ARCHIVE_NAME}"
echo ""
echo "Содержимое архива:"
unzip -l "${ARCHIVE_NAME}"
