"""Show and save a board's log from its ST-LINK virtual COM port (115200 baud).

    python tools/serial_log.py list           the COM ports, and which board each one belongs to
    python tools/serial_log.py dk             the STM32N6570-DK's log, also saved to logs/dk.txt
    python tools/serial_log.py wba            the NUCLEO-WBA55CG's log, also saved to logs/wba.txt
    python tools/serial_log.py COM7 [file]    a port by name

Stop with Ctrl+C. Every line is printed and saved with the PC's time in front. Needs pyserial
(pip install pyserial). A board is found by matching its ST-LINK serial number (from
STM32_Programmer_CLI, as flash.py does) with the COM port's USB serial number.
"""
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOARDS = {"dk": "STM32N6570", "wba": "WBA55"}


def ports():
    """[(device, usb serial number, description)] of the serial ports."""
    from serial.tools import list_ports
    return [(p.device, p.serial_number or "", p.description or "") for p in list_ports.comports()]


def board_of_serials():
    """{ST-LINK serial number: board name}, empty if STM32_Programmer_CLI is not there."""
    try:
        sys.path.insert(0, HERE)
        import flash
        return {sn: name for sn, name in flash.probes()}
    except Exception:
        return {}


def find_port(board):
    names = board_of_serials()
    for device, sn, _ in ports():
        if sn and BOARDS[board].lower() in names.get(sn, "").lower():
            return device
    sys.exit("No COM port found for the %s: run 'python tools/serial_log.py list', then give the COM port" % board)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    what = sys.argv[1]
    if what == "list":
        names = board_of_serials()
        for device, sn, desc in ports():
            print("%-8s %-28s %s" % (device, names.get(sn, "?"), desc))
        return
    import serial
    if what.lower() in BOARDS:
        port = find_port(what.lower())
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "logs", what.lower() + ".txt")
    else:
        port = what
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "logs", port.replace("/", "_") + ".txt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print("%s at 115200 baud, saving to %s (Ctrl+C to stop)" % (port, out), flush=True)
    with serial.Serial(port, 115200, timeout=0.5) as s, open(out, "a", encoding="utf-8") as f:
        f.write("\n--- %s %s ---\n" % (datetime.datetime.now().isoformat(timespec="seconds"), port))
        try:
            while True:
                raw = s.readline()
                if not raw:
                    continue
                line = "%s  %s" % (datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                   raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
        except KeyboardInterrupt:
            print("\nsaved to " + out)


if __name__ == "__main__":
    main()
