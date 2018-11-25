# !/usr/bin/env python
# encoding: utf-8
import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    print('Connected')
    mqtt.subscribe('hermes/#')
    mqtt.publish("hermes/hotword/default/detected")

def on_publish(client, userdata, flags, rc):
    print('Connected')
    mqtt.subscribe('hermes/#')
    mqtt.publish("hermes/hotword/default/detected")

def on_message(client, userdata, msg):
    # Parse the json response
    intent_json = json.loads(msg.payload)
    intentName = intent_json['intent']['intentName']
    slots = intent_json['slots']
    print('Intent {}'.format(intentName))
    for slot in slots:
        slot_name = slot['slotName']
        raw_value = slot['rawValue']
        value = slot['value']['value']
        print('Slot {} -> \n\tRaw: {} \tValue: {}'.format(slot_name, raw_value, value))

client1 = mqtt.Client()
client1.on_connect = on_connect
client1.on_message = on_message
client1.on_publish = on_publish
client1.on_publish = on_publish
client1.connect('snips-london-2.local', 1883)
ret= client1.publish("hermes/hotword/default/detected","default", qos=2, retain=True)
client1.loop_forever()
