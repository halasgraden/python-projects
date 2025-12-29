import random

import string

from hangman_random_words import word_list

#Generates a random word
def get_word():
    return random.choice(word_list)

#Starting condition
start = print(
        "Your objective is to guess the five letter word in 6 guesses. Good luck!"
    )

#Prompts user to repeat game
def repeat_game():
    repeat = input("Would you like to play again? (Y/N): ")
    if repeat == "Y":
        play_game()
    else:
        exit()

#Game is active
def play_game():
    word = get_word()
    blank_word = "_" * len(word)
    guessed = False
    guessed_letters = []
    attempts = 6
    while not guessed and attempts > 0:
        guess = input("Please guess a letter: ").lower().strip()
        if len(guess) == 1 and guess in string.ascii_lowercase:
            if guess in guessed_letters:
                print("You have already guessed that letter. Please try again.")
            elif guess not in word:
                attempts -= 1
                guessed_letters.append(guess)
                print(f"Sorry, '{guess}' is not in the word.")
            else:
                guessed_letters.append(guess)
                word_as_list = list(blank_word)
                indices = [i for i, letter in enumerate(word) if letter == guess]
                for index in indices:
                    word_as_list[index] = guess
                blank_word = "".join(word_as_list)
                if "_" not in blank_word:
                    guessed = True
        else:
            print("That is not a valid response, please try again.")
        print(blank_word)
        print(f"Attempts left: {attempts}")
    if guessed == True:
        print(f"You got it! The correct word was {word}")
        repeat_game()
    else:
        #Loss condition
        print(f"Sorry, you ran out of guesses. The correct word was {word}.")
        repeat_game()

def main():
    play_game()

if __name__ == "__main__":
    main()