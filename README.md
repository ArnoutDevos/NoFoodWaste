# Reducing Food Waste with Privacy Preserving Computer Vision
LauzHack 2018

IBM api:
`pip install --upgrade "watson-developer-cloud>=2.4.1"`

Idea/goals:
- [x] Every 30 seconds: take 4K picture on *client*
- [x] blur people on client
- [x] send blurred and patched versions from *client* to *server* (= increased security/anonymity)
- [x] Server does API call to IBM Watson Vision to classify as food/non-food
- [x] Make snips.ai board talk with you, that you should go and check out food in #building. You can then ask what food is present.
- [ ] Store results in database: (timestamp, building, {11: "food", 12: "non-food", ...}). keys are array indexes ij of patches
- [ ] For those patches that come back with "food" or "fruit" or ..., send them to the IBM Watson Vision "food" API to determine the type of food. Store this too.
- [ ] Count people
