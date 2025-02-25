# running_time,correct_tokens_front
import random as rn
import sys
import serial

# set to true to use the local simulation which creates a random result based on the seed
simulate = False
correct_tokens = 0
logfilename = 'out.log'

temporary_result = ""


def interpret_input(input: list[str]):
    assert (len(input) == 2)
    #print(f"Seed was {int(input[0])}")
    rn.seed(int(input[0]))
    i = 0
    while i < int(input[1]):
        rn.randint(0, 999999999)
        i += 1


def run_via_serial(input: str):
    global temporary_result
    device = serial.Serial(port='/dev/ttyUSB3', baudrate=9600)
    # inputstring = ",".join([str(x) for x in input]) + \
    #    '\n'  # \n-terminate the input
    #print(f"Input string is: \"{inputstring}\"")
    device.write(bytes(input, 'ascii'))
    output = device.readline()  # the output needs to be \n-terminated
    device.close()
    temporary_result += ", running time: " + str(output, 'UTF-8').strip()
    f = open(logfilename, 'a')
    f.write(temporary_result + "\n")
    f.close()
    return str(output, 'UTF-8').strip()


def createPWD(seed: int, offset: int):
    global temporary_result
    global correct_tokens
    rn.seed(seed)
    i = 0
    while i < offset:
        rn.randint(0, 999999999)
        i += 1
    res = rn.randint(100000000, 999999999)
    # todo convert to a string of symbols later?
    # collect number of correct tokens (until the first one?)
    stringrep = str(res)

    temporary_result = "pwd: " + stringrep + ", seed: " + \
        str(seed) + ", offset: " + str(offset)

    for i in range(9):
        if stringrep[i] == str(i+1):
            correct_tokens += 1
        else:
            break
    #print(f"Generated pwd: {res} has {correct_tokens} correct tokens.")
    temporary_result += ", correct_digits: " + str(correct_tokens)
    return stringrep + "\n"


if __name__ == "__main__":
    # interpret_input(sys.argv[1:])
    if simulate:
        res = rn.random()
        print(f"{res}")
    else:
        print(
            f"{run_via_serial(createPWD(int(sys.argv[1]), int(sys.argv[2])))},{correct_tokens}")
