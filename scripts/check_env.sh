#!/usr/bin/env bash
# Verifica se o ambiente de desenvolvimento está completo para PSoC6 dual-core.
# Uso: bash scripts/check_env.sh
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[OK]${NC}    $*"; }
fail() { echo -e "  ${RED}[MISS]${NC}  $*"; ((FAIL_COUNT++)) || true; }
warn() { echo -e "  ${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${CYAN}>>> $*${NC}"; }

FAIL_COUNT=0

# --------------------------------------------------------------------------- #
section "ARM GNU Toolchain (Cortex-M0+ / M4)"

check_bin() {
    local bin="$1"; local label="${2:-$1}"
    if cmd=$(command -v "$bin" 2>/dev/null); then
        ver=$(${bin} --version 2>/dev/null | head -1 || echo "?")
        ok "$label  →  $cmd  ($ver)"
    else
        fail "$label não encontrado no PATH"
    fi
}

check_bin arm-none-eabi-gcc     "arm-none-eabi-gcc  (compilador C)"
check_bin arm-none-eabi-g++     "arm-none-eabi-g++  (compilador C++)"
check_bin arm-none-eabi-objcopy "arm-none-eabi-objcopy"
check_bin arm-none-eabi-size    "arm-none-eabi-size"
check_bin arm-none-eabi-nm      "arm-none-eabi-nm"

# gdb pode ser gdb-multiarch em sistemas Debian/Ubuntu
if command -v arm-none-eabi-gdb &>/dev/null; then
    check_bin arm-none-eabi-gdb "arm-none-eabi-gdb"
elif command -v gdb-multiarch &>/dev/null; then
    ver=$(gdb-multiarch --version 2>/dev/null | head -1)
    ok "gdb-multiarch  (alias para ARM)  →  $(command -v gdb-multiarch)  ($ver)"
else
    fail "arm-none-eabi-gdb / gdb-multiarch não encontrado"
fi

# --------------------------------------------------------------------------- #
section "Flash / Debug (OpenOCD + KitProg3)"

check_bin openocd "openocd"

# Verifica interface KitProg3 nos scripts do OpenOCD
OPENOCD_SCRIPTS=$(openocd --help 2>&1 | grep -oP '(?<=scripts: ).*' | head -1 || true)
if [ -z "$OPENOCD_SCRIPTS" ]; then
    OPENOCD_SCRIPTS=$(find /usr /opt "$HOME" -maxdepth 8 -name "kitprog3.cfg" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
fi
if [ -n "$OPENOCD_SCRIPTS" ] && find "$OPENOCD_SCRIPTS" -name "kitprog3.cfg" &>/dev/null 2>&1; then
    ok "Interface kitprog3.cfg encontrada em $OPENOCD_SCRIPTS"
elif find /usr /opt "$HOME" -maxdepth 10 -name "kitprog3.cfg" &>/dev/null 2>&1; then
    KP3=$(find /usr /opt "$HOME" -maxdepth 10 -name "kitprog3.cfg" 2>/dev/null | head -1)
    ok "Interface kitprog3.cfg encontrada: $KP3"
else
    warn "kitprog3.cfg não localizado — pode precisar do OpenOCD do ModusToolbox"
fi

# --------------------------------------------------------------------------- #
section "Build System"

check_bin cmake   "cmake"
check_bin ninja   "ninja"
check_bin make    "make"

# --------------------------------------------------------------------------- #
section "ModusToolbox 3.x"

MTB_DEFAULT=""
for candidate in /opt/Tools/ModusToolbox/tools_* "$HOME/ModusToolbox/tools_*" "$HOME/ModusToolbox"; do
    if [ -d "$candidate" ]; then
        MTB_DEFAULT="$candidate"
        break
    fi
done

if [ -n "$MTB_DEFAULT" ]; then
    ok "Diretório ModusToolbox: $MTB_DEFAULT"
    # Versão
    MTB_VER_FILE=$(find "$MTB_DEFAULT" -maxdepth 3 -name "version.xml" 2>/dev/null | head -1)
    if [ -n "$MTB_VER_FILE" ]; then
        ver=$(grep -oP '(?<=<version>)[^<]+' "$MTB_VER_FILE" 2>/dev/null || echo "?")
        ok "Versão MTB: $ver"
    fi
    # GCC bundled
    MTB_GCC=$(find "$MTB_DEFAULT" -name "arm-none-eabi-gcc" 2>/dev/null | head -1)
    if [ -n "$MTB_GCC" ]; then
        ver=$("$MTB_GCC" --version 2>/dev/null | head -1)
        ok "GCC bundled MTB: $MTB_GCC  ($ver)"
    else
        warn "GCC bundled não encontrado dentro de $MTB_DEFAULT"
    fi
    # project-creator-cli
    if command -v project-creator-cli &>/dev/null; then
        ok "project-creator-cli disponível"
    else
        warn "project-creator-cli não está no PATH"
    fi
else
    fail "ModusToolbox não instalado em /opt/Tools/ModusToolbox ou \$HOME/ModusToolbox"
    echo -e "       ${YELLOW}→ Baixe em: https://www.infineon.com/modustoolbox${NC}"
fi

# --------------------------------------------------------------------------- #
section "Python + dependências offline"

PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python3}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null)}"

if [ -n "$PYTHON_BIN" ] && "$PYTHON_BIN" --version &>/dev/null; then
    ver=$("$PYTHON_BIN" --version 2>&1)
    ok "Python: $PYTHON_BIN  ($ver)"
else
    fail "python3 não encontrado"
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
    ok "Virtualenv ativo: $VIRTUAL_ENV"
else
    warn "Nenhum virtualenv ativo — ative com: source .venv/bin/activate"
fi

PY_PKG_LABELS=(numpy torch scikit-learn wfdb PyYAML pandas matplotlib)
PY_PKG_MODULES=(numpy torch sklearn wfdb yaml pandas matplotlib)
for i in "${!PY_PKG_MODULES[@]}"; do
    pkg="${PY_PKG_LABELS[$i]}"
    module="${PY_PKG_MODULES[$i]}"
    if "$PYTHON_BIN" -c "import $module" 2>/dev/null; then
        ver=$("$PYTHON_BIN" -c "import $module; print(getattr($module,'__version__','?'))" 2>/dev/null)
        ok "Python: $pkg  ($ver)"
    else
        fail "Python: $pkg não instalado"
    fi
done

# --------------------------------------------------------------------------- #
section "USB / Serial (para KitProg3 e UART)"

if command -v lsusb &>/dev/null; then
    ok "lsusb disponível"
    if lsusb 2>/dev/null | grep -qi "04b4"; then
        ok "Dispositivo Infineon/Cypress detectado no USB"
    else
        warn "Nenhum dispositivo Infineon (VID 04b4) conectado no momento"
    fi
else
    fail "usbutils (lsusb) não instalado"
fi

if groups | grep -q "dialout"; then
    ok "Usuário no grupo dialout (acesso /dev/ttyACM*)"
else
    warn "Usuário NÃO está no grupo dialout"
    warn "  → sudo usermod -aG dialout \$USER  (requer logout/login)"
fi

if groups | grep -q "plugdev"; then
    ok "Usuário no grupo plugdev (acesso USB HID)"
else
    warn "Usuário NÃO está no grupo plugdev"
    warn "  → sudo usermod -aG plugdev \$USER  (requer logout/login)"
fi

# Regra udev KitProg3
if [ -f "/etc/udev/rules.d/99-infineon-kitprog.rules" ]; then
    ok "Regra udev KitProg3: /etc/udev/rules.d/99-infineon-kitprog.rules"
else
    warn "Regra udev KitProg3 ausente — execute scripts/setup_toolchain.sh"
fi

# --------------------------------------------------------------------------- #
section "Resumo"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "\n  ${GREEN}Ambiente completo — nenhum item crítico faltando.${NC}\n"
else
    echo -e "\n  ${RED}$FAIL_COUNT item(s) crítico(s) faltando. Veja [MISS] acima.${NC}"
    echo -e "  ${YELLOW}Para instalar: bash scripts/setup_toolchain.sh${NC}\n"
fi

exit "$FAIL_COUNT"
