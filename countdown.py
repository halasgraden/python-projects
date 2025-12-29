import time

#Prompting user
user_time = int(input("Please provide your time in seconds: "))

#If wrong user input, tell user
if user_time < 0:
    raise ValueError("Please provide a correct number of seconds.")

if not isinstance(user_time, int):
    print("Please enter an integer.")

#Countdown timer in descending fashion
for n in range(user_time, 0, -1):
    seconds = n % 60
    minutes = n // 60
    print(f"{minutes:02}:{seconds:02}")
    time.sleep(1)

print("Time is up.")