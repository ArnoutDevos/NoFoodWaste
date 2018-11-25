#!/usr/bin/env python2
import requests
from hermes_python.hermes import Hermes
import random

MQTT_IP_ADDR = "localhost"
MQTT_PORT = 1883
MQTT_ADDR = "{}:{}".format(MQTT_IP_ADDR, str(MQTT_PORT))

INTENT_FOOD = "arnoutdevos:ask_food"
INTENT_INTEREST = "arnoutdevos:ask_more_info"
INTENT_SATISFIED = "arnoutdevos:thank_you"
INTENT_STOP = "arnoutdevos:stop_info"

INTENT_FILTER_GET_ANSWER = [
    INTENT_STOP,
    INTENT_INTEREST,
    INTENT_SATISFIED
]

INTENT_FILTER_GET_FINAL_ANSWER = [
    INTENT_STOP,
    INTENT_SATISFIED
]

API_HOST = 'http://localhost:8000'

SessionsStates = {}


def user_ask_food(hermes, intent_message):
    session_id = intent_message.session_id

    try:
        r = requests.get('/food-watch/get-food-available')
        r.raise_for_status()
        r_data = r.json()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as ex:
        r_data = {}
        print(ex)

    food = r_data.get('food')
    if food:
        response = "Hey buddy, I saw some free food in the BC building. Don't let it go to waste!"
    else:
        response = "No food"

    hermes.publish_continue_session(session_id, response, INTENT_FILTER_GET_ANSWER)


def user_more_info(hermes, intent_message):
    session_id = intent_message.session_id

    list_foods = ["pizza", "beer"]

    specific_food = None
    if intent_message.slots is not None:
        specific_food = intent_message.slots.answer.first().value
        if specific_food in list_foods:
            response = "It is your lucky day, they have {}.".format(specific_food)
            hermes.publish_end_session(session_id, response)
        else:
            response = "Unfortunately, they don't have any free {0} today, but they do have {1}.".format(specific_food, str(list_foods))
            hermes.publish_end_session(session_id, response)

    else:
        response = "On today's menu they serve {}.".format(list_foods)

    hermes.publish_continue_session(intent_message.session_id, response, INTENT_FILTER_GET_FINAL_ANSWER)


def user_quits(hermes, intent_message):
    session_id = intent_message.session_id

    # clean up
    #del SessionsStates[session_id]
    response = "That's unfortunate, my man. Your day will come."

    hermes.publish_end_session(session_id, response)

def user_satisfied(hermes, intent_message):
    session_id = intent_message.session_id

    # clean up
    #del SessionsStates[session_id]
    response = "Alright, enjoy your meal!"

    hermes.publish_end_session(session_id, response)

with Hermes(MQTT_ADDR) as h:

    h.subscribe_intent(INTENT_FOOD, user_ask_food) \
        .subscribe_intent(INTENT_STOP, user_quits) \
        .subscribe_intent(INTENT_INTEREST, user_more_info) \
        .subscribe_intent(INTENT_SATISFIED, user_satisfied) \
        .start()
