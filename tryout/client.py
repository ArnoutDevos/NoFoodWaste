import paho.mqtt.client as mqtt

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    #client.subscribe("hermes/hotword/#")
    #client.subscribe("hermes/asr/#")
    #client.subscribe('hermes/dialogueManager/#')
    #client.subscribe('hermes/hotword/default/#')
    client.publish('hermes/hotword/default/detected','{"siteId":"default","modelId":"hey_snips","modelVersion":"hey_snips_3.1_2018-04-13T15:27:35_model_0019","modelType":"universal","currentSensitivity":0.5}', retain=True)
    #client.publish('hermes/dialogManager/default/startSession')
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("snips-london-2.local", 1883, 60)
#client.startSession()
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
