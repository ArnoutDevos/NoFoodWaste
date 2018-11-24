# NoFoodWaste
LauzHack 2018

IBM api:
`pip install --upgrade "watson-developer-cloud>=2.4.1"`

Logitech keyboard:
`pip install pyusb`

- Every 1 minute: take 4K picture on *client*
- Count people, blur people on client and send patched versions from *client* to *server* (= increased security/anonymity)
- Server does API call to IBM Watson Vision to classify as food/non-food, and stores database: (timestamp, building, {11: "food", 12: "non-food", ...} (keys are array indexes ij of patches)
- For those patches that come back with "food" or "fruit" or ..., send them to the IBM Watson Vision "food" API to determine the type of food. Store this too.
- if there are >= 5 of the most recent entries (different timestamps) of a patch at the same location with "food", and there are >= XXX patches in such a "food" situation, a call for hungry students is sent out
- (make this sending out of call for hungry students depending on the amount of people present)
- (Make snips.ai board talk to you, that you should go and check out food in <building>. You can then ask what food is present.
