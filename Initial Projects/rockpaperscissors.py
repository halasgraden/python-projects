import random

#Defining variable
def play():
    user = input("Enter your choice: ")
    user = user.lower()

    computer = random.choice(["rock", "paper", "scissors"])

    if you_win(user, computer):
        return "You have chosen {} and the computer has chosen {}, you win!".format(user, computer)

    if user == computer:
        return "You and the computer have both chosen {}. It's a draw.".format(computer)

    return "You have chosen {} and the computer has chosen {}, you lost.".format(user, computer)

#Defining win/lose
def you_win(player, opponent):
    if (player == "rock" and opponent == "scissors") or (player == "paper" and opponent == "rock") or (player == "scissors" and opponent == "paper"):
        return True
    return False

if __name__ == '__main__':
    print(play())