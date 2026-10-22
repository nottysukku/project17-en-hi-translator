"""Flash the Project 17 boards: the NUCLEO-WBA55CG music receiver and the STM32N6570-DK translator.

    python flash.py list                 show the connected ST-LINK probes
    python flash.py wba                  build output -> NUCLEO-WBA55CG internal flash
    python flash.py n6                   sign and program the N6 firmware (FSBL + app)
    python flash.py n6 --models          also program every model blob (ASR, translation, voice)

Both boards may be plugged in at once: each command picks its probe by the board name the ST-LINK
reports. The N6 must be in development boot mode while it is programmed (BOOT1 = H), then back in
flash boot mode (BOOT1 = L) to run; the script waits for the switch when it is needed.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # N6_AutoSpeech
CUBEPROG = os.environ.get("CUBE_PROGRAMMER_BIN",
                          r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin")
CLI = os.path.join(CUBEPROG, "STM32_Programmer_CLI.exe")
SIGN = os.path.join(CUBEPROG, "STM32_SigningTool_CLI.exe")
N6_LOADER = os.path.join(CUBEPROG, "ExternalLoader", "MX66UW1G45G_STM32N6570-DK.stldr")

WBA_BIN = os.path.join(ROOT, "wba_music_receiver", "build", "wba_music_receiver.bin")
N6_REPO = os.path.join(ROOT, "Reverse_engineering2", "Reverse_Engineering", "stm32n6-speech-recognition")
N6_BUILD = os.path.join(N6_REPO, "build_translator")
N6_FSBL_BIN = os.path.join(N6_REPO, "fsbl", "build", "application", "fsbl.bin")
# Model blobs and their NOR addresses; written by the model tools next to the firmware.
N6_MODELS_JSON = os.path.join(N6_REPO, "src", "application", "translator", "models.json")


def run(cmd):
    print("> " + " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("command failed with exit code %d" % result.returncode)


def probes():
    """[(serial, board name)] of the connected ST-LINKs."""
    out = subprocess.run([CLI, "-l", "st-link"], capture_output=True, text=True).stdout
    found = []
    serial = None
    for line in out.splitlines():
        m = re.search(r"ST-LINK SN\s*:\s*(\S+)", line)
        if m:
            serial = m.group(1)
            found.append([serial, ""])
        m = re.search(r"Board Name\s*:\s*(.+)", line)
        if m and found:
            found[-1][1] = m.group(1).strip()
    return [tuple(p) for p in found]


def pick(fragment):
    """Serial of the probe whose board name contains fragment."""
    found = probes()
    if not found:
        sys.exit("No ST-LINK found: is the board's ST-LINK USB port connected?")
    match = [s for s, name in found if fragment.lower() in name.lower()]
    if len(match) == 1:
        return match[0]
    if not match and len(found) == 1 and not found[0][1]:
        return found[0][0]  # one probe that reports no name: assume it is the right one
    sys.exit("Cannot tell which probe is the %s; connected: %s" % (fragment, found))


def flash_wba(_args):
    if not os.path.exists(WBA_BIN):
        sys.exit("Missing %s: build wba_music_receiver first" % WBA_BIN)
    serial = pick("WBA55")
    run([CLI, "-c", "port=SWD", "sn=" + serial, "mode=UR", "-d", WBA_BIN, "0x08000000", "-v", "-rst"])
    print("\nNUCLEO-WBA55CG programmed. In the phone's Bluetooth settings, pair with N6_Music_xxxx.")


def sign_n6():
    for src, dst in ((N6_FSBL_BIN, os.path.join(N6_BUILD, "fsbl-trusted.bin")),
                     (os.path.join(N6_BUILD, "src", "application", "app.bin"),
                      os.path.join(N6_BUILD, "app-trusted.bin"))):
        if not os.path.exists(src):
            sys.exit("Missing %s: build the firmware first" % src)
        run([SIGN, "-s", "-bin", src, "-nk", "-of", "0x80000000", "-t", "fsbl", "-o", dst, "-hv", "2.3", "-align"])


def flash_n6(args):
    sign_n6()
    serial = pick("STM32N6570")
    if not args.yes:
        input("Set BOOT1 (SW1) on the DK to H (right), press RESET, then press Enter here... ")
    cmd = [CLI, "-c", "port=SWD", "sn=" + serial, "freq=24000", "ap=1", "-el", N6_LOADER,
           "-d", os.path.join(N6_BUILD, "fsbl-trusted.bin"), "0x70000000",
           "-d", os.path.join(N6_BUILD, "app-trusted.bin"), "0x70100000"]
    if args.models:
        with open(N6_MODELS_JSON, encoding="utf-8") as f:
            for blob in json.load(f):
                path = os.path.join(N6_REPO, blob["file"])
                if not os.path.exists(path):
                    sys.exit("Missing model blob %s" % path)
                cmd += ["-d", path, blob["address"]]
    run(cmd + ["-v"])
    print("\nN6 programmed. Set BOOT1 (SW1) back to L (left) and press RESET to run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="target", required=True)
    sub.add_parser("list")
    sub.add_parser("wba")
    n6 = sub.add_parser("n6")
    n6.add_argument("--models", action="store_true", help="also program the model blobs")
    n6.add_argument("--yes", action="store_true", help="do not wait for the BOOT1 switch")
    args = parser.parse_args()
    if args.target == "list":
        for serial, name in probes():
            print(serial, name or "(no board name)")
    elif args.target == "wba":
        flash_wba(args)
    else:
        flash_n6(args)


if __name__ == "__main__":
    main()
