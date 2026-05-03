import json
import socket
import argparse


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def generate(station_id: str, port: int, output: str) -> None:
    try:
        import qrcode
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Install dulu: pip install qrcode[pil] pillow")
        return

    ip = get_local_ip()

    # Payload yang di-encode ke QR
    # Mobile app akan parse JSON ini setelah scan
    payload = json.dumps({
        "station_id": station_id,
        "ip": ip,
        "port": port,
    })

    print(f"IP yang terdeteksi : {ip}")
    print(f"Station ID         : {station_id}")
    print(f"QR Payload         : {payload}")

    # Generate QR
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#FF6500", back_color="white")

    # Tambah label di bawah QR
    base = img.convert("RGB")
    w, h = base.size
    label_h = 60
    final = Image.new("RGB", (w, h + label_h), "white")
    final.paste(base, (0, 0))

    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    text = f"EnerGym Station: {station_id}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, h + 10), text, fill="#FF6500", font=font)
    draw.text(((w - text_w) // 2 + 10, h + 32), f"IP: {ip}:{port}", fill="#888", font=font)

    final.save(output)
    print(f"\nQR berhasil dibuat: {output}")
    print("Cetak file ini dan tempel di alat gym.")
    print("\nPERINGATAN: Pastikan HP dan laptop di WiFi yang SAMA saat latihan!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-id", default="STATION_01")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default="qr_station.png")
    args = parser.parse_args()

    generate(args.station_id, args.port, args.output)
