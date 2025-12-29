import re
import ast
from langchain_core.tools import BaseTool, Tool
import subprocess

def extract_json_from_output(output: str):
    try:
        # Look for the last valid JSON object
        lines = output.split('\n')
        if len(lines) >= 1:
            required_data = lines[-1]
            required_data = ast.literal_eval(required_data)

            return required_data
        else:
            print('No valid data found!')
    except Exception as e:
        print(e)
        return {}

def run_sdk_command(mode: list):
    """Runs the Advantech SDK CLI wrapper with the given mode."""
    mode = ' '.join(mode)
    result = subprocess.run(
        ["script", "-q", "-c", f"python3 edgesync_utils.py {mode}", "/dev/null"],
        capture_output=True,
        text=True
    )
    print(result)

    if result.returncode != 0:
        return {"error": f"SDK crashed (code {result.returncode})"}

    try:
        return extract_json_from_output(result.stdout.strip())
    except Exception:
        return {"error": "Failed to parse output"}


def get_board_info(input_data: dict) -> str:
    output = run_sdk_command(["info"])
    lines = ["\n"]
    for src, temp in output.items():
        lines.append(f"{src}: {temp}")
    return "\n".join(lines)

def get_voltages(input_data: dict) -> str:
    output = run_sdk_command(["voltages"])
    lines = ["\n"]
    for src,temp in output.items():
        lines.append(f"{src}: {temp} V")
    return "\n".join(lines)

def get_temperatures(input_data: dict) -> str:
    output = run_sdk_command(["temperatures"])
    lines = ["\n"]
    for src,temp in output.items():
        lines.append(f"{src}: {temp} °C")
    return "\n".join(lines)



def get_fans(input_data: dict):
    output = run_sdk_command(["fans"])
    if "error" in output:
        return f"\nError: {output['error']}"
    lines = ["\n"]
    for src, val in output.items():
        lines.append(f"{src}: {val} RPM")
    return "\n".join(lines)


def get_gpio_overview(input_data: dict):
    output = run_sdk_command(["gpio"])
    if "error" in output:
        return f"\nError: {output['error']}"
    lines = ["\n"]
    for pin, details in output.items():
        direction = details.get("direction", "Unknown")
        level = details.get("level", "Unknown")
        lines.append(f"{pin}: Direction={direction}, Level={level}")
    return "\n".join(lines)


def set_gpio_pin(input_data: dict):
    """
    Example input_data: "pin=GPIO1, level=HIGH"
    """
    try:
        pin = input_data.get("tool_pin",None)
        level = input_data.get("tool_level", None).upper()

        if not pin:
            return "\nError: Missing pin parameter."

        output = run_sdk_command(["gpio_set", pin, level])
        if "error" in output:
            return f"\nError: {output['error']}"

        return f"\nPin {pin} set to {level}. Result: {output.get('set_level_result', 'Unknown')}"
    except Exception as e:
        return f"\nError parsing input for gpio_set: {e}"


def read_gpio_pin(input_data: dict):
    """
    Example input_data: "pin=GPIO1"
    """
    try:
        pin = input_data.get("tool_pin",None)

        if not pin:
            return "\nError: Missing pin parameter."

        output = run_sdk_command(["gpio_read", pin])
        if "error" in output:
            return f"\nError: {output['error']}"

        return f"\nPin {pin} input level: {output.get('input_level', 'Unknown')}"
    except Exception as e:
        return f"\nError parsing input for gpio_read: {e}"


device_info = Tool(
    name="device_info_tool",
    func=get_board_info,
    description="Use this to retrieve detailed motherboard and BIOS information, such as manufacturer, model, BIOS version, and library version."
)

device_voltage = Tool(
    name="device_voltage_tool",
    func=get_voltages,
    description="Use this to get real-time voltage readings from all onboard voltage sources of the device."
)

device_temperature = Tool(
    name="device_temperature_tool",
    func=get_temperatures,
    description="Use this to fetch current temperature data from all onboard temperature sensors on the device."
)

device_fans = Tool(
    name="device_fans_tool",
    func=get_fans,
    description="Use this to check real-time fan speed (RPM) readings for each onboard fan sensor."
)

device_gpio_overview = Tool(
    name="gpio_pins_overview",
    func=get_gpio_overview,
    description="Use this to get an overview of GPIO pin directions and logic levels."
)

gpio_set_tool = Tool(
    name="gpio_set_tool",
    func=set_gpio_pin,
    description="Use this to set the output level of a GPIO pin. Provide input as: 'pin=GPIO1, level=HIGH'"
)

gpio_read_tool = Tool(
    name="gpio_read_tool",
    func=read_gpio_pin,
    description="Use this to read the input level of a GPIO pin. Provide input as: 'pin=GPIO1'"
)

tools = [device_info, device_voltage, device_temperature, device_fans, device_gpio_overview, gpio_set_tool, gpio_read_tool]