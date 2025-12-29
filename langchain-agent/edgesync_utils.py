import json
import advantech
import sys

def get_device():
    """Create and return the Advantech device instance."""
    try:
        return advantech.edge.Device()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize device: {e}")


def get_board_info(device=None):
    device = device
    platform_info = device.platform_information

    if not platform_info.motherboard_name:
        return {"error": "Platform information feature is not supported on this device."}

    info = {
        "Motherboard": platform_info.motherboard_name,
        # "Manufacturer": platform_info.motherboard_manufacturer,
        "BIOS Revision": platform_info.bios_revision,
        "Driver Version": platform_info.driver_version or "Not Available",
        "Library Version": platform_info.library_version or "Not Available",
        "Firmware Name": getattr(platform_info, "firmware_name", "Not Available"),
        "Firmware Version": getattr(platform_info, "firmware_version", "Not Available"),
        "Boot-up Times": getattr(platform_info, "boot_up_times", "Not Available"),
        "Running Time (hours)": getattr(platform_info, "running_time_in_hours", "Not Available"),
    }
    print(info)
    return info


def get_temperatures(device=None):
    device = device
    sensors = device.onboard_sensors

    try:
        temps = {}
        for source in sensors.temperature_sources:
            try:
                temps[source] = sensors.get_temperature(source)
            except Exception as e:
                temps[source] = f"Error: {e}"

        if len(temps)==0:
            print({"error": "No Temperature source available"})
        print(temps)
        return temps
    except Exception as e:
        return {"error": f"Temperature sensors not supported: {e}"}


def get_voltages(device=None):
    device = device or advantech.edge.Device()
    sensors = device.onboard_sensors

    try:
        volts = {}
        for source in sensors.voltage_sources:
            try:
                volts[source] = sensors.get_voltage(source)
            except Exception as e:
                volts[source] = f"Error: {e}"

        if len(volts)==0:
            print({"error": "No Voltage source available"})
            return {"error": "No Voltage source available"}
        print(volts)
        return volts
    except Exception as e:
        return {"error": f"Voltage sensors not supported: {e}"}


def get_fan_speeds(device=None):
    device = device
    sensors = device.onboard_sensors

    try:
        if not hasattr(sensors, "get_fan_speed"):
            return {"error": "Fan speed monitoring not implemented."}

        fans = {}
        for source in sensors.fan_sources:
            try:
                fans[source] = sensors.get_fan_speed(source)
            except Exception as e:
                fans[source] = f"Error: {e}"
        if len(fans)==0:
            print({"error": "No fan source available"})
            return {"error": "No fan source available"}
        print(fans)
        return fans
    except Exception as e:
        return {"error": f"Fan speed sensors not supported: {e}"}


def get_gpio_overview(device=None):
    device = device or advantech.edge.Device()
    gpio = device.gpio

    try:
        pins_info = {}
        for pin in gpio.pin_names:
            direction = gpio.get_direction(pin).name
            level = gpio.get_level(pin).name
            pins_info[pin] = {"direction": direction, "level": level}

        if len(pins_info)==0:
            print({"error": "No Pins available"})
            return {"error": "No Pins available"}
        print(pins_info)
        return pins_info
    except Exception as e:
        return {"error": f"GPIO feature not supported: {e}"}


def set_gpio_output(pin_name, level=advantech.edge.ifeatures.igpio.GpioLevelTypes.High, device=None):
    device = device or advantech.edge.Device()
    gpio = device.gpio

    try:
        gpio.set_direction(pin_name, advantech.edge.ifeatures.igpio.GpioDirectionTypes.Output)
        if type(level)==str:
            level = level.title()
        level = advantech.edge.ifeatures.igpio.GpioLevelTypes[level]
        result = gpio.set_level(pin_name, level)
        print({"pin": pin_name, "set_level_result": result})
        return {"pin": pin_name, "set_level_result": result}
    except Exception as e:
        print({"error": f"Failed to read input level for {pin_name}: {e}"})
        return {"error": f"Failed to set output level for {pin_name}: {e}"}


def read_gpio_input(pin_name, device=None):
    device = device or advantech.edge.Device()
    gpio = device.gpio

    try:
        # gpio.set_direction(pin_name, advantech.edge.ifeatures.igpio.GpioDirectionTypes.Input)
        input_level = gpio.get_level(pin_name)
        print({"pin": pin_name, "input_level": input_level.name})
        return {"pin": pin_name, "input_level": input_level}
    except Exception as e:
        return {"error": f"Failed to read input level for {pin_name}: {e}"}



def main():
    device = get_device()
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No mode specified"}))
        return

    mode = sys.argv[1]

    try:
        if mode == "temperatures":
            return get_temperatures(device)
        elif mode == "voltages":
            return get_voltages(device)
        elif mode == "info":
            return get_board_info(device)
        elif mode == "fans":
            return get_fan_speeds(device)
        elif mode == "gpio":
            return get_gpio_overview(device)
        elif mode == "gpio_set":
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Missing pin name or level for gpio_set"}))
                return
            pin = sys.argv[2]
            level = sys.argv[3]
            result = set_gpio_output(pin, level, device)
        elif mode == "gpio_read":
            if len(sys.argv) < 3:
                print(json.dumps({"error": "Missing pin name for gpio_read"}))
                return
            pin = sys.argv[2]
            result = read_gpio_input(pin, device)
        else:
            result = {"error": f"Unknown mode: {mode}"}
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()