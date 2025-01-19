import boto3
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from sklearn.preprocessing import normalize
from scipy.stats import unitary_group
from braket.circuits.result_types import Probability

# --- 1. Text to Vector Embedding (using pre-trained GloVe embeddings) ---

def text_to_vector(text, glove_embeddings):
    """
    Converts text to a vector embedding using GloVe.

    Args:
        text (str): The input text.
        glove_embeddings (dict): A dictionary mapping words to GloVe vectors.

    Returns:
        np.ndarray: The vector embedding of the text.
    """
    tokens = text.lower().split()  # Simple tokenization
    vectors = [glove_embeddings.get(token, np.zeros(50)) for token in tokens]  # Assume 50-dim GloVe
    if not vectors:
        return np.zeros(50)
    return np.mean(vectors, axis=0)  # Average word embeddings


# --- 2. Vector to Quantum State (Amplitude Encoding) using Unitary---
def create_amplitude_encoding_circuit(vector):
    """
    Creates a quantum circuit for amplitude encoding using a unitary matrix.

    Args:
        vector (np.ndarray): The normalized vector to encode.

    Returns:
        braket.circuits.Circuit: The quantum circuit.
    """

    normalized_vector = normalize(vector.reshape(1, -1), norm='l2')[0]
    num_qubits = int(np.ceil(np.log2(len(normalized_vector))))

    # Pad the vector with zeros to the nearest power of 2
    padded_vector = np.pad(normalized_vector, (0, 2 ** num_qubits - len(normalized_vector)))

    # Generate a unitary matrix that maps |0...0> to the desired state
    # using Householder transformation
    v = np.zeros(2 ** num_qubits)
    v[0] = 1.0

    u = padded_vector - v
    u = u / np.linalg.norm(u)

    unitary_matrix = np.eye(2 ** num_qubits) - 2.0 * np.outer(u, u)

    # Create a circuit and apply the unitary
    circ = Circuit()
    q = list(range(num_qubits))
    circ.unitary(matrix=unitary_matrix, targets=q)

    return circ
# def create_amplitude_encoding_circuit(vector):
#     """
#     Creates a quantum circuit for amplitude encoding using a unitary matrix.
#
#     Args:
#         vector (np.ndarray): The normalized vector to encode.
#
#     Returns:
#         braket.circuits.Circuit: The quantum circuit.
#     """
#
#     normalized_vector = normalize(vector.reshape(1, -1), norm='l2')[0]
#     num_qubits = int(np.ceil(np.log2(len(normalized_vector))))
#
#     # Pad the vector with zeros to the nearest power of 2
#     padded_vector = np.pad(normalized_vector, (0, 2 ** num_qubits - len(normalized_vector)))
#
#     # Generate a unitary matrix that maps |0...0> to the desired state
#     unitary_matrix = np.eye(2 ** num_qubits)
#     unitary_matrix[0] = padded_vector
#
#     # Create a circuit and apply the unitary
#     circ = Circuit()
#     q = list(range(num_qubits))
#     circ.unitary(matrix=unitary_matrix, targets=q)
#
#     return circ


# --- 3. QKD (Simplified Simulation) ---

def generate_qkd_key(length):
    """
    Generates a random bit string to simulate a QKD key.

    Args:
        length (int): The desired length of the key.

    Returns:
        str: The random key string.
    """
    return ''.join(str(np.random.randint(0, 2)) for _ in range(length))


# --- 4. QOTP Encryption ---

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


# --- 5. QOTP Decryption ---

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
            decryption_circ.x(i)  # Apply X gates based on the key (same as encryption)

    return decryption_circ


# --- 6. Measurement and Decoding ---

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

    # Add measurement to the circuit using result types
    num_qubits = decryption_circuit.qubit_count
    for qubit in range(num_qubits):
        decryption_circuit.add_result_type(Probability(target=[qubit]))

    # Run the circuit
    task = device.run(decryption_circuit, shots=shots)
    result = task.result()

    # Get the probabilities from the result
    probabilities = result.values[0]

    # Reconstruct the vector from probabilities
    reconstructed_vector = np.sqrt(probabilities)

    # Pad the reconstructed vector with zeros to match the GloVe vector dimension
    padding_length = len(original_vector) - len(reconstructed_vector)
    if padding_length > 0:
        reconstructed_vector = np.pad(reconstructed_vector, (0, padding_length))

    # Calculate cosine similarity
    similarities = {
        word: np.dot(reconstructed_vector, vector) / (np.linalg.norm(reconstructed_vector) * np.linalg.norm(vector))
        for word, vector in glove_embeddings.items()}

    closest_word = max(similarities, key=similarities.get)

    return closest_word


# --- Main Program ---

if __name__ == "__main__":
    # Load pre-trained GloVe embeddings (download a GloVe file and load it)
    glove_embeddings = {}
    with open("/Users/gouthamarcot/Documents/personal/codebase/Quantum/glove/glove.6B.200d.txt", "r", encoding="utf-8") as f:  # Replace with your GloVe file
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_embeddings[word] = vector

    # Example text
    text = "This is a secret message"

    # 1. Text to Vector
    text_vector = text_to_vector(text, glove_embeddings)

    # 2. Vector to Quantum State
    encoding_circuit = create_amplitude_encoding_circuit(text_vector)

    # 3. QKD
    key_length = encoding_circuit.qubit_count  # You can adjust the key length
    key = generate_qkd_key(key_length)

    # 4. Encryption
    encryption_circuit = create_encryption_circuit(encoding_circuit, key)

    # 5. Decryption
    decryption_circuit = create_decryption_circuit(encryption_circuit, key)

    # Choose a device (local simulator or a real quantum device)
    # device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")  # Example: SV1 simulator
    device = LocalSimulator()

    # 6. Measurement and Decoding
    shots = 1000
    decoded_text = measure_and_decode(decryption_circuit, shots, device, glove_embeddings, text_vector)

    print(f"Original Text: {text}")
    print(f"Decoded Text: {decoded_text}")
