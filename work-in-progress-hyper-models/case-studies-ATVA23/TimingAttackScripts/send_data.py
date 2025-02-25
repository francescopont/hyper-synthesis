# running_time
import random as rn
import sys
import serial


def run_via_serial(input: list):
    device = serial.Serial(port='/dev/ttyUSB0', baudrate=9600)
    inputstring = ",".join([str(x) for x in input]) + \
        '\n'  # \n-terminate the input
    #print(f"Input string is: \"{inputstring}\"")
    device.write(bytes(inputstring, 'ascii'))
    output = device.readline()  # the output needs to be \n-terminated
    device.close()
    return str(output)


if __name__ == "__main__":
    print(f"{run_via_serial(sys.argv[1:])}")
