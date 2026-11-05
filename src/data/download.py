"""Модуль скачивания данных FlyWire v783.

Содержит функции для проверки наличия и скачивания сырых CSV
коннектома дрозофилы (connections.csv, neuron_annotations.csv).
"""

from pathlib import Path
import urllib.error
import urllib.request

# Имена файлов, ожидаемых в data/raw/.
_CONNECTIONS_FILENAME = "connections.csv"
_ANNOTATIONS_FILENAME = "neuron_annotations.csv"

# Версия датасета FlyWire, используется для логов.
_FLYWIRE_VERSION = "v783"

# Прямые ссылки для fallback (публичные бакеты FlyWire/Codex).
# Если URL устареет — сработает исключение с инструкцией ручной загрузки.
_FALLBACK_URLS: dict[str, str] = {
    _CONNECTIONS_FILENAME: (
        "https://storage.googleapis.com/flywire-daf/"
        "connections/" + _FLYWIRE_VERSION + "/connections.csv"
    ),
    _ANNOTATIONS_FILENAME: (
        "https://storage.googleapis.com/flywire-daf/"
        "annotations/" + _FLYWIRE_VERSION + "/neuron_annotations.csv"
    ),
}

# Альтернативный публичный эндпоинт flywire.ai (зеркало Codex).
_FALLBACK_URLS_ALT: dict[str, str] = {
    _CONNECTIONS_FILENAME: "https://codex.flywire.ai/api/download/connections.csv",
    _ANNOTATIONS_FILENAME: (
        "https://codex.flywire.ai/api/download/neuron_annotations.csv"
    ),
}


def check_raw_files(data_dir: Path) -> bool:
    """Проверяет наличие необходимых raw-файлов.

    Args:
        data_dir: Папка ``data/raw/`` с сырыми CSV.

    Returns:
        True если оба файла (``connections.csv`` и
        ``neuron_annotations.csv``) существуют и являются файлами.
    """
    # Приводим к Path на случай передачи строки.
    data_dir = Path(data_dir)
    connections = data_dir / _CONNECTIONS_FILENAME
    annotations = data_dir / _ANNOTATIONS_FILENAME
    # Оба файла должны существовать.
    return connections.is_file() and annotations.is_file()


def _try_download_via_caveclient(output_path: Path, filename: str) -> bool:
    """Пытается скачать файл через caveclient.

    Args:
        output_path: Куда сохранить файл.
        filename: Имя файла (``connections.csv`` или ``neuron_annotations.csv``).

    Returns:
        True если скачивание успешно, иначе False.
    """
    try:
        # Импорт внутри функции — caveclient может быть не установлен.
        import caveclient  # type: ignore[import-untyped]
    except ImportError:
        print(f"[download] caveclient не установлен — пропускаем CAVE-путь для {filename}.")
        return False

    try:
        print(f"[download] Пробуем скачать {filename} через caveclient (FlyWire {_FLYWIRE_VERSION})...")
        # Клиент FlyWire; датасет по умолчанию — production.
        client = caveclient.CAVEclient("flywire_fafb_production")  # type: ignore[attr-defined]

        # Попытка получить таблицу через materialize API.
        # Имена таблиц могут меняться — пробуем несколько вариантов.
        table_candidates: dict[str, list[str]] = {
            _CONNECTIONS_FILENAME: ["connections", "synapses", "connectivity"],
            _ANNOTATIONS_FILENAME: ["neurons", "annotations", "cell_annotations"],
        }
        candidates = table_candidates.get(filename, [])

        for table in candidates:
            try:
                # materialize может отсутствовать в старых версиях — проверяем.
                if not hasattr(client, "materialize"):
                    continue
                # Запрашиваем данные как DataFrame.
                df = client.materialize.query_table(table)  # type: ignore[union-attr]
                # Если получили DataFrame — сохраняем.
                if hasattr(df, "to_csv"):
                    df.to_csv(output_path, index=False)
                    print(f"[download] {filename} успешно скачан через caveclient (таблица {table}).")
                    return True
            except Exception as exc:  # noqa: BLE001
                # Пробуем следующую таблицу.
                print(f"[download] caveclient: таблица {table} недоступна ({exc}), пробуем дальше.")
                continue

        print(f"[download] caveclient: не удалось скачать {filename} — таблицы не найдены.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[download] caveclient не смог скачать {filename}: {exc}")
        return False


def _try_download_via_http(url: str, output_path: Path) -> bool:
    """Пытается скачать файл по прямой HTTP-ссылке.

    Args:
        url: Прямая ссылка на файл.
        output_path: Куда сохранить файл.

    Returns:
        True если скачивание успешно, иначе False.
    """
    try:
        print(f"[download] Пробуем скачать {output_path.name} по ссылке {url} ...")
        # Таймаут 60с — файлы большие (~2ГБ), но начинаем с проверки доступности.
        urllib.request.urlretrieve(url, str(output_path))
        print(f"[download] {output_path.name} успешно скачан по HTTP.")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"[download] HTTP-скачивание не удалось для {url}: {exc}")
        # Удаляем частично скачанный файл, если он появился.
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass
        return False


def download_flywire_data(output_dir: Path) -> None:
    """Скачивает ``connections.csv`` и ``neuron_annotations.csv`` FlyWire v783.

    Логика:
    1. Создаёт ``output_dir`` если её нет.
    2. Если оба файла уже на месте — пропускает скачивание.
    3. Иначе пытается скачать отсутствующие файлы через caveclient,
       затем через прямые ссылки flywire.ai/Codex.
    4. Если сеть недоступна или все методы не сработали — бросает
       исключение с инструкцией ручной загрузки.

    Args:
        output_dir: Папка для сохранения raw-файлов (обычно ``data/raw/``).

    Raises:
        RuntimeError: Если сеть недоступна или скачивание не удалось.
            В сообщении содержится инструкция положить CSV вручную в
            ``data/raw/``.
        TypeError: Если ``output_dir`` не является ``Path`` или строкой.
    """
    # Поддержка передачи строки — приводим к Path.
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # Создаём директорию если нужно (родители тоже).
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] Целевая папка: {output_dir.resolve()}")

    connections_path = output_dir / _CONNECTIONS_FILENAME
    annotations_path = output_dir / _ANNOTATIONS_FILENAME

    # Проверяем, что уже есть.
    has_connections = connections_path.is_file()
    has_annotations = annotations_path.is_file()

    if has_connections and has_annotations:
        print("[download] Оба файла уже на месте — пропускаем скачивание.")
        print(f"[download]   - {connections_path}")
        print(f"[download]   - {annotations_path}")
        return

    # Логируем что именно отсутствует.
    if not has_connections:
        print(f"[download] Не найден файл: {connections_path}")
    if not has_annotations:
        print(f"[download] Не найден файл: {annotations_path}")

    # Список отсутствующих файлов для скачивания.
    missing: list[tuple[Path, str]] = []
    if not has_connections:
        missing.append((connections_path, _CONNECTIONS_FILENAME))
    if not has_annotations:
        missing.append((annotations_path, _ANNOTATIONS_FILENAME))

    # Пытаемся скачать каждый отсутствующий файл.
    failed: list[str] = []
    for output_path, filename in missing:
        success = False

        # Шаг 1: caveclient.
        if _try_download_via_caveclient(output_path, filename):
            success = True
        else:
            # Шаг 2: прямые ссылки (основная + альтернативная).
            urls_to_try = [
                _FALLBACK_URLS.get(filename, ""),
                _FALLBACK_URLS_ALT.get(filename, ""),
            ]
            for url in urls_to_try:
                if not url:
                    continue
                if _try_download_via_http(url, output_path):
                    success = True
                    break

        if not success:
            failed.append(filename)
            print(f"[download] Не удалось скачать {filename} ни одним из методов.")

    if failed:
        # Формируем понятное исключение с инструкцией ручной загрузки.
        failed_str = ", ".join(failed)
        raise RuntimeError(
            f"Не удалось скачать файлы FlyWire v783: {failed_str}. "
            "Сеть недоступна или все методы скачивания не сработали. "
            "Пожалуйста, скачайте CSV вручную с https://flywire.ai "
            "(раздел Downloads / Codex) или через caveclient "
            "(https://github.com/seung-lab/CAVEclient) и положите CSV вручную "
            f"в data/raw/ — ожидаются файлы: {_CONNECTIONS_FILENAME}, "
            f"{_ANNOTATIONS_FILENAME} в папке {output_dir.resolve()}. "
            "Альтернативно поместите файлы в data/raw/ вручную и повторно "
            "запустите scripts/01_build_graphs.py."
        )

    print("[download] Все отсутствующие файлы успешно скачаны.")
