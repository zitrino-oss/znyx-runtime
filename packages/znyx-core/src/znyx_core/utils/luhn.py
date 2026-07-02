def luhn_check(card_number: str) -> bool:
    """
    Validate a card number using the Luhn algorithm.

    Args:
        card_number: String of digits representing a card number

    Returns:
        True if valid according to Luhn algorithm, False otherwise
    """
    # Remove any spaces or dashes
    card_number = card_number.replace(' ', '').replace('-', '')

    # Must be all digits
    if not card_number.isdigit():
        return False

    # Must be at least 13 digits (minimum valid card length)
    if len(card_number) < 13:
        return False

    # Luhn algorithm
    total = 0
    reverse_digits = card_number[::-1]

    for i, digit in enumerate(reverse_digits):
        n = int(digit)

        # Double every second digit
        if i % 2 == 1:
            n *= 2
            # If result is two digits, add them together
            if n > 9:
                n -= 9

        total += n

    # Valid if total is divisible by 10
    return total % 10 == 0
