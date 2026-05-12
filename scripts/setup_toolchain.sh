#!/usr/bin/env bash
# Instala toolchain ARM + ferramentas de build para PSoC6 dual-core no Linux.
# Uso: bash scripts/setup_toolchain.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${YELLOW}=== $* ===${NC}"; }

# --------------------------------------------------------------------------- #
section "1. Dependências apt (ARM toolchain, OpenOCD, build tools)"

PKGS=(
    gcc-arm-none-eabi          # compilador C/C++ para Cortex-M
    binutils-arm-none-eabi     # assembler, linker, objcopy, size
    gdb-multiarch              # debugger (cobre ARM; arm-none-eabi-gdb é alias)
    openocd                    # flash/debug via KitProg3 (CMSIS-DAP / J-Link)
    cmake                      # sistema de build (usado pelo MTB e projetos bare-metal)
    ninja-build                # backend de build rápido
    libncurses5                # dependência do GDB em algumas distros
    libusb-1.0-0               # KitProg3 USB HID/bulk
    usbutils                   # lsusb para diagnóstico
    minicom                    # terminal serial (alternativa ao miniterm)
    python3-serial             # pyserial para python -m serial.tools.miniterm
)

MISSING=()
for pkg in "${PKGS[@]}"; do
    dpkg -s "$pkg" &>/dev/null || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -eq 0 ]; then
    info "Todos os pacotes apt já instalados."
else
    info "Instalando: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING[@]}"
fi

# --------------------------------------------------------------------------- #
section "2. Regra udev para KitProg3 / CY8CKIT (acesso USB sem root)"

UDEV_FILE="/etc/udev/rules.d/99-infineon-kitprog.rules"
if [ -f "$UDEV_FILE" ]; then
    info "Regra udev já existe: $UDEV_FILE"
else
    info "Criando regra udev para KitProg3..."
    sudo tee "$UDEV_FILE" > /dev/null <<'UDEV'
# Infineon KitProg3 / PSoC 6 Kits
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", ATTR{idProduct}=="f150", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", ATTR{idProduct}=="f151", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", ATTR{idProduct}=="f152", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="tty", ATTRS{idVendor}=="04b4", MODE="0666", GROUP="dialout"
UDEV
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    info "Regra criada. Adicione seu usuário ao grupo plugdev e dialout se ainda não estiver:"
    warn "  sudo usermod -aG plugdev,dialout \$USER  (requer logout/login)"
fi

# --------------------------------------------------------------------------- #
section "3. ModusToolbox 3.x (requer download manual)"

MTB_DEFAULT="$HOME/ModusToolbox"
if [ -d "$MTB_DEFAULT" ]; then
    info "ModusToolbox encontrado em $MTB_DEFAULT"
    MTB_GCC=$(find "$MTB_DEFAULT" -name "arm-none-eabi-gcc" 2>/dev/null | head -1)
    if [ -n "$MTB_GCC" ]; then
        MTB_GCC_BIN=$(dirname "$MTB_GCC")
        info "GCC bundled do MTB: $MTB_GCC_BIN"
        if ! echo "$PATH" | grep -q "$MTB_GCC_BIN"; then
            warn "Adicione ao ~/.bashrc:"
            warn "  export PATH=\"$MTB_GCC_BIN:\$PATH\""
        fi
    fi
else
    warn "ModusToolbox NÃO encontrado em $MTB_DEFAULT"
    warn "Download manual necessário:"
    warn "  1. Acesse: https://www.infineon.com/modustoolbox"
    warn "  2. Baixe o instalador Linux (.run) da versão 3.x"
    warn "  3. Execute:"
    warn "       chmod +x ModusToolbox_3.x.x.xxxx-linux-install.run"
    warn "       ./ModusToolbox_3.x.x.xxxx-linux-install.run"
    warn "  4. Adicione ao ~/.bashrc:"
    warn "       export PATH=\$HOME/ModusToolbox/tools_3.x/gcc/bin:\$PATH"
    warn "       export MTB_TOOLS_DIR=\$HOME/ModusToolbox/tools_3.x"
fi

# --------------------------------------------------------------------------- #
section "Concluído"
info "Execute o script de verificação para checar o ambiente:"
info "  bash scripts/check_env.sh"
