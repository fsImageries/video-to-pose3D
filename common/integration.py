from pyUtils import nphelper
from pyUtils import pyhelper


def check_outgoings(args, data):
    send_maya = getattr(args, "send_maya", None)

    if send_maya:
        send_to_maya(data)


def send_to_maya(data):
    client = pyhelper.ClientBase()
    try:
        client.connect()
        client.send(data, json_cls=nphelper.NumpyEncoder)
        print("[LOG] Successfully send rotation to maya.")
    except Exception as err:
        print("[ERROR] Error occured:\n{0}".format(err))
    finally:
        client.disconnect()
