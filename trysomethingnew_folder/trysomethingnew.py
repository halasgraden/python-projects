#The goal of this is to give me a task, skill, experience, or activity to try or learn every week
#I think curious people (and those wanting uncertainty) would love to have an improved and interactive version of this
#It should use Google, Yelp, and other APIs to derive things to do and import them into a list here
#Gamify this by implementing a score system (figure out how to keep track of this long-term) [Have achievements]
#Use a GUI and basic HTML/CSS to create a visual?
#Consider Email (smtplib), SMS (Twilio API), Push notifications (desktop or mobile) [AUTOMATE IT TO PROMPT YOU THEN SEND IT TO YOU ON SUNDAY?]

import random
from ideabank import idea_list, theme_list

def main():
    match = find_random_match()
    if match != None:
        store_chosen_matches(match)

def load_stored_matches():
    try:
        with open("stored_matches.txt") as file:
            lines = file.readlines()
            stored_ideas = [line.strip() for line in lines]
            return stored_ideas
    except FileNotFoundError:
        return []

def get_user_preferences():
    budget = input("Free, low, medium, or high cost? (free/low/medium/high): ").strip().lower()
    remote_inperson = input("Remote or in person? (remote/inperson): ").strip().lower()
    time_commitment = input("Short (1-2 hours), medium(3-4 hours), or long (6-10 hours/full day)? (short/medium/long): ").strip().lower()
    return {
            "budget": budget,
            "remote_inperson": remote_inperson,
            "time_commitment": time_commitment
            }

def print_idea(idea):
    print(f"\nTry this: {idea["idea"]}")
    print(f"Theme: {idea["theme"]}")
    print(f"Preferences:")
    for key, value in idea["preferences"].items():
        print(f" - {key}: {value}")

def store_chosen_matches(match):
    with open("stored_matches.txt", "a") as file:
        file.write(f"{match["idea"]}\n")

def find_random_match():
    stored_ideas = load_stored_matches()
    unused_ideas = [idea for idea in idea_list if idea["idea"] not in stored_ideas]
    if not unused_ideas:
        print("You've tried everything! Please add more ideas.")
        return None
    
    print(f"\npreferences  themes  chance")
    user_pick = input(f"Please pick (type) one: ").strip().lower()
    if user_pick == "preferences":
        prefs = get_user_preferences()
        matches = [
            idea for idea in unused_ideas
            if all(
                key in idea["preferences"] and idea["preferences"][key] == value
                for key, value in prefs.items()
            )
        ]
        if matches:
            chosen = random.choice(matches)
            print_idea(chosen)
            return chosen
        else:
            print("No matches found.")
            return find_random_match()

    elif user_pick == "chance":
        chosen = random.choice(unused_ideas)
        print_idea(chosen)
        return chosen

    elif user_pick == "themes":
        print("\n" + ", ".join(theme_list))
        choose_theme = input("Please choose (type) a theme: ")
        if choose_theme in theme_list and not stored_ideas:
            matches = [idea for idea in unused_ideas if idea["theme"] == choose_theme]
            chosen = random.choice(matches)
            print_idea(chosen)
            return chosen
        else:
            print("Please choose a correct theme.")
            return find_random_match()

    else:
        print("Please try again.")
        find_random_match()

if __name__ == "__main__":
    main()