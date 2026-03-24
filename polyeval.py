import math

def horner(coeffs, x):
    r = coeffs[-1]
    levels = 0
    for c in coeffs[-2::-1]:
        r = r * x + c
        levels += 1
    return r, levels


def all_storage_peterson_stockmeyer(coeffs, x):
    if len(coeffs) == 1:
        return coeffs[0], 0
    r = coeffs[0] + coeffs[1] * x
    x_pws = [1.0, x]
    pws_levels = [0, 0]
    coeffs_levels = [0, 1]
    for i in range(2, len(coeffs)):
        high_pow = 2**math.floor(math.log2(i)) - 1
        low_pow = i - high_pow
        x_pws.append(x_pws[low_pow] * x_pws[high_pow])
        pws_levels.append(max(pws_levels[low_pow], pws_levels[high_pow]) + 1)
        coeffs_levels.append(max(pws_levels[low_pow] + 1, pws_levels[high_pow]) + 1)
        r += (coeffs[i] * x_pws[low_pow]) * x_pws[high_pow]
    return r, max(coeffs_levels)

def twospws_storage_peterson_stockmeyer(coeffs, x):
    i = 0
    curr_x = x
    pows = []
    pows_levels = []
    while 2 ** i < len(coeffs):
        pows.append(curr_x)
        pows_levels.append(i)
        curr_x = curr_x * curr_x
        i += 1

    r = 0.0
    coeffs_levels = []
    for i, c in enumerate(coeffs):
        curr_x = None
        curr_pow_idx = 0
        curr_pow_level = 0
        while i > 0:
            if i % 2 == 1:
                if curr_x is None:
                    curr_x = c * pows[curr_pow_idx]
                    curr_pow_level = pows_levels[curr_pow_idx] + 1
                else:
                    curr_x = curr_x * pows[curr_pow_idx]
                    curr_pow_level = max(curr_pow_level, pows_levels[curr_pow_idx]) + 1
            curr_pow_idx += 1
            i = i // 2

        if curr_x is None:
            r += c
        else:
            r += curr_x
        coeffs_levels.append(curr_pow_level)
    return r, max(coeffs_levels)

def computation_only_peterson_stockmeyer(coeffs, x):
    r = 0.0
    coeffs_levels = []
    for i, c in enumerate(coeffs):
        curr_x = None
        running_x = x
        running_level = 0
        curr_pow_idx = 0
        curr_pow_level = 0
        while i > 0:
            if i % 2 == 1:
                if curr_x is None:
                    curr_x = c * running_x
                    curr_pow_level = running_level + 1
                else:
                    curr_x = curr_x * running_x
                    curr_pow_level = max(curr_pow_level, running_level) + 1
            running_x = running_x * running_x
            running_level += 1
            curr_pow_idx += 1
            i = i // 2
        if curr_x is None:
            r += c
        else:
            r += curr_x
        coeffs_levels.append(curr_pow_level)
    return r, max(coeffs_levels)
        

if __name__ == "__main__":
    coeffs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = 2
    print(horner(coeffs, x))
    print(all_storage_peterson_stockmeyer(coeffs, x))
    print(twospws_storage_peterson_stockmeyer(coeffs, x))
    print(computation_only_peterson_stockmeyer(coeffs, x))