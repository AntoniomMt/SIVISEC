import os
import argparse
import shutil
import random
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm

# ultralytics
try:
    from ultralytics import YOLO
except Exception:
    print("ERROR: ultralytics no encontrado. Instala con: pip install ultralytics")
    raise


# -------------------------------------------------------------------
# Cargar lista de clases
# -------------------------------------------------------------------
def read_classes(classes_path: Path):
    if not classes_path.exists():
        return None
    with classes_path.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


# -------------------------------------------------------------------
# Preparar dataset: dividir en train/val y copiar
# -------------------------------------------------------------------
def prepare_dataset(images_dir: Path, labels_dir: Path, out_dir: Path, seed=42, train_ratio=0.8):

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    labels = sorted([p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"])

    # nombres sin extensión
    image_names = {p.stem: p for p in images}
    label_names = {p.stem: p for p in labels}

    common = sorted(list(image_names.keys() & label_names.keys()))
    if len(common) == 0:
        raise RuntimeError("No se encontraron archivos con nombres coincidentes entre images/ y labels/.")

    pairs = [(image_names[name], label_names[name]) for name in common]

    # shuffle reproducible
    random.seed(seed)
    random.shuffle(pairs)

    # split
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # crear estructura de carpetas
    train_img = out_dir / "images" / "train"
    val_img = out_dir / "images" / "val"
    train_lbl = out_dir / "labels" / "train"
    val_lbl = out_dir / "labels" / "val"

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # copiar archivos
    for imgp, lblp in tqdm(train_pairs, desc="Copiando train"):
        shutil.copy2(imgp, train_img / imgp.name)
        shutil.copy2(lblp, train_lbl / lblp.name)

    for imgp, lblp in tqdm(val_pairs, desc="Copiando val"):
        shutil.copy2(imgp, val_img / imgp.name)
        shutil.copy2(lblp, val_lbl / lblp.name)

    return train_img, val_img, train_lbl, val_lbl


# -------------------------------------------------------------------
# Escribir data.yaml
# -------------------------------------------------------------------
def write_data_yaml(out_dir: Path, classes):
    yaml_path = out_dir / "data.yaml"
    data = {
        "train": str((out_dir / "images" / "train").resolve().as_posix()),
        "val": str((out_dir / "images" / "val").resolve().as_posix()),
        "names": {i: c for i, c in enumerate(classes)}
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    return yaml_path


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="./SUPER/images")
    parser.add_argument("--labels", type=str, default="./SUPER/labels")
    parser.add_argument("--classes", type=str, default="./SUPER/classes.txt")
    parser.add_argument("--out", type=str, default="./dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--project", type=str, default="runs/train_custom")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise RuntimeError("La carpeta images/ no existe.")
    if not labels_dir.exists():
        raise RuntimeError("La carpeta labels/ no existe.")

    # leer classes.txt
    classes = read_classes(Path(args.classes))
    if classes is None:
        print("No se encontró classes.txt; usando clases por defecto: ['bag','person']")
        classes = ["bag", "person"]

    print("\n=== Preparando dataset ===")
    prepare_dataset(images_dir, labels_dir, out_dir, train_ratio=args.train_ratio)

    print("\n=== Generando data.yaml ===")
    data_yaml = write_data_yaml(out_dir, classes)
    print("data.yaml creado en:", data_yaml)

    # device automático
    if args.device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except:
            device = "cpu"
    else:
        device = args.device

    print(f"\n=== Iniciando entrenamiento YOLO (device={device}) ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name if args.name else f"exp_{timestamp}"

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=exp_name,
        device=device,
        exist_ok=True
    )

    print("\n=== ENTRENAMIENTO FINALIZADO ===")
    print("Revisa: runs/train_custom/", exp_name)


if __name__ == "__main__":
    main()
