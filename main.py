import boto3
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from sklearn.preprocessing import normalize

def text_to_vector(text, glove_embeddings):
    """
    Converts text to a vector embedding using GloVe.

    Args:
        text (str): The input text.
        glove_embeddings (dict): A dictionary mapping words to GloVe vectors.

    Returns:
        np.ndarray: The vector embedding of the text.
    """
    tokens = text.lower().split()
    vectors = [glove_embeddings.get(token, np.zeros(50)) for token in tokens]  # Assume 50-dim GloVe
    if not vectors:
        return np.zeros(50)
    return np.mean(vectors, axis=0)


def create_amplitude_encoding_circuit(vector):
    """
    Creates a quantum circuit for amplitude encoding.

    Args:
        vector (np.ndarray): The normalized vector to encode.

    Returns:
        braket.circuits.Circuit: The quantum circuit.
    """
    
    normalized_vector = normalize(vector.reshape(1, -1), norm='l2')[0]
    num_qubits = int(np.ceil(np.log2(len(normalized_vector))))    
    padded_vector = np.pad(normalized_vector, (0, 2**num_qubits - len(normalized_vector)))
    circ = Circuit()
    circ.initialize(padded_vector)
    return circ


def generate_qkd_key(length):
    """
    Generates a random bit string to simulate a QKD key.

    Args:
        length (int): The desired length of the key.

    Returns:
        str: The random key string.
    """
    return ''.join(str(np.random.randint(0, 2)) for _ in range(length))


def create_encryption_circuit(encoding_circuit, key):
    """
    Applies QOTP encryption to the encoded state.

    Args:
        encoding_circuit (braket.circuits.Circuit): The circuit that encodes the data.
        key (str): The QKD key.

    Returns:
        braket.circuits.Circuit: The encryption circuit.
    """
    num_qubits = encoding_circuit.qubit_count
    encryption_circ = Circuit()
    encryption_circ.add_circuit(encoding_circuit)

    for i in range(min(num_qubits, len(key))):
        if key[i] == '1':
            encryption_circ.x(i)
    return encryption_circ


def create_decryption_circuit(encrypted_circuit, key):
    """
    Applies QOTP decryption.

    Args:
        encrypted_circuit (braket.circuits.Circuit): The encrypted circuit.
        key (str): The QKD key.

    Returns:
        braket.circuits.Circuit: The decryption circuit.
    """
    num_qubits = encrypted_circuit.qubit_count
    decryption_circ = Circuit()
    decryption_circ.add_circuit(encrypted_circuit)
    
    for i in range(min(num_qubits, len(key))):
        if key[i] == '1':
            decryption_circ.x(i) # Apply X gates based on the key (same as encryption)

    
    return decryption_circ


def measure_and_decode(decryption_circuit, shots, device, glove_embeddings, original_vector):
    """
    Measures the decrypted state, reconstructs the vector, and finds the closest words.

    Args:
        decryption_circuit (braket.circuits.Circuit): The decryption circuit.
        shots (int): The number of measurement shots.
        device (AwsDevice): The quantum device or simulator.
        glove_embeddings (dict): The GloVe embeddings dictionary.
        original_vector (np.ndarray): original embedding vector.

    Returns:
        str: The decoded text (approximation).
    """
    
    
    decryption_circuit.measure_all()
    task = device.run(decryption_circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts

    reconstructed_vector = np.zeros(len(original_vector))
    for bitstring, count in counts.items():
        index = int(bitstring, 2)
        if index < len(reconstructed_vector):
            reconstructed_vector[index] += count

    reconstructed_vector = reconstructed_vector / shots  

    similarities = {word: np.dot(reconstructed_vector, vector) / (np.linalg.norm(reconstructed_vector) * np.linalg.norm(vector))
                    for word, vector in glove_embeddings.items()}

    closest_word = max(similarities, key=similarities.get)

    return closest_word


if __name__ == "__main__":
    glove_embeddings = {}
    with open("glove.6B.50d.txt", "r", encoding="utf-8") as f: 
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_embeddings[word] = vector

    text = "This is a secret message"

    text_vector = text_to_vector(text, glove_embeddings)

    encoding_circuit = create_amplitude_encoding_circuit(text_vector)
    key_length = encoding_circuit.qubit_count # You can adjust the key length
    key = generate_qkd_key(key_length)
    encryption_circuit = create_encryption_circuit(encoding_circuit, key)
    decryption_circuit = create_decryption_circuit(encryption_circuit, key)
    # device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")  # Example: SV1 is a noiseless simulator
    device = LocalSimulator()

    shots = 1000 #ideal
    decoded_text = measure_and_decode(decryption_circuit, shots, device, glove_embeddings, text_vector)

    print(f"Original Text: {text}")
    print(f"Decoded Text: {decoded_text}")
