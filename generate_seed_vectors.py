import random

f = open("./random_seed_vectors.txt", "w")

# generate independent random vectors for features
for line in range(0, 68+68):
    for d in range(0, 10000):
        # Generate random element with 50% probability of being 0 or 1
        element = random.randrange(0,2)
        f.write(str(element))
    f.write("\n")

# Generate similar vectors for levels
ln_random_vector = []
for d in range(0, 10000):
    element = random.randrange(0, 2)
    ln_random_vector.append(element)
    f.write(str(element))

f.write("\n")

# Flip D/m bits each time or 10000/200 = 50
for line in range(0, 200):
    bits_to_flip = random.sample(range(0,10000), 50)
    for bit_index in bits_to_flip:
        if ln_random_vector[bit_index] == 0:
            ln_random_vector[bit_index] = 1
        else:
            ln_random_vector[bit_index] = 0

    for element in ln_random_vector:
        f.write(str(element))

    f.write("\n")

f.close()

