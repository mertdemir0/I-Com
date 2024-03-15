import random

def create_deck():
    """Returns a shuffled deck of cards (list of tuples)."""
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
    deck = [(value, suit) for suit in suits for value in values]
    random.shuffle(deck)
    return deck

def card_value(card):
    """Returns the value of a single card (int)."""
    value, suit = card
    if value in ['Jack', 'Queen', 'King']:
        return 10
    elif value == 'Ace':
        return 11
    else:
        return int(value)

def hand_value(hand):
    """Returns the total value of a hand of cards."""
    value = sum(card_value(card) for card in hand)
    num_aces = sum(1 for card in hand if card[0] == 'Ace')
    
    # Adjust Aces from 11 to 1 if total value exceeds 21
    while value > 21 and num_aces:
        value -= 10
        num_aces -= 1
    
    return value

def play_blackjack():
    deck = create_deck()
    player_hand = [deck.pop(), deck.pop()]
    dealer_hand = [deck.pop(), deck.pop()]
    
    # Player's turn
    while hand_value(player_hand) < 21:
        print("Your hand:", player_hand, "Value:", hand_value(player_hand))
        action = input("Do you want to (h)it or (s)tand? ")
        if action.lower() == 'h':
            player_hand.append(deck.pop())
        elif action.lower() == 's':
            break
    
    # Dealer's turn
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(deck.pop())
    
    # Show hands
    print("Your final hand:", player_hand, "Value:", hand_value(player_hand))
    print("Dealer's final hand:", dealer_hand, "Value:", hand_value(dealer_hand))
    
    # Determine the winner
    player_value = hand_value(player_hand)
    dealer_value = hand_value(dealer_hand)
    if player_value > 21 or (dealer_value <= 21 and dealer_value > player_value):
        print("Dealer wins!")
    elif dealer_value > 21 or dealer_value < player_value:
        print("You win!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_blackjack()
