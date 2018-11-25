import paho.mqtt.client as mqtt

HOST = 'snips-london-2.local'
PORT = 1883

def on_connect(client, userdata, flags, rc):
    print("Connected to {0} with result code {1}".format(HOST, rc))
    # Subscribe to the hotword detected topic
    client.subscribe("hermes/hotword/default/detected")
    # Subscribe to intent topic
    client.subscribe('hermes/intent/INTENT_NAME')

def on_message(client, userdata, msg):
    if msg.topic == 'hermes/hotword/default/detected':
        print("Hotword detected!")
    elif msg.topic == 'hermes/intent/INTENT_NAME':
        print("Intent detected!")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.publish("hermes/hotword/default/detected", payload="default")
client.connect(HOST, PORT, 60)
client.loop_forever()
