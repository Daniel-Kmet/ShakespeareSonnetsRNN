import numpy as np
import time
import platform
import psutil

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize RNN parameters and hyperparameters.
        
        Args:
            input_size (int): Size of input vector (vocabulary size)
            hidden_size (int): Size of hidden state vector
            output_size (int): Size of output vector (vocabulary size)
            learning_rate (float): Learning rate for gradient descent
        """
        # Initialize weights with small random values to break symmetry
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases to zero
        self.bz = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        
    def forward(self, inputs, z_prev):
        """
        Forward pass through the RNN.
        
        Args:
            inputs: List/array of input vectors, each of shape (input_size,) or (input_size, 1)
            z_prev: Initial hidden state of shape (hidden_size, 1)
            
        Returns:
            tuple: (output vector from final time step, final hidden state)
            
        Notes:
            - Stores intermediate values in self.xs, self.zs, and self.os for use in backward pass.
            - Uses tanh activation for hidden state and softmax for output.
        """
        self.xs = []
        self.zs = []
        self.os = []
        z = z_prev
        for x in inputs:
            # x is a column vector
            x = x.reshape(-1, 1)
            self.xs.append(x)

            # hidden state: z_t = tanh(Wx * x_t + Wz * z_prev + bz)
            z = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wz, z) + self.bz)
            self.zs.append(z)

            # output: y_t = Wy * z_t + by
            y = np.dot(self.Wy, z) + self.by

            # softmax : o_t = softmax(y_t)
            o = self.softmax(y)
            self.os.append(o)
        return self.os[-1], z

    def backward(self, target):
        """
        Backward pass through time (BPTT).
        
        Args:
            target: One-hot encoded target vector (vocabulary size,)
            
        Notes:
            - Implements backpropagation through time over all time steps.
            - Calculates gradients for Wx, Wz, Wy, bz, and by using stored forward pass values.
            - Updates weights and biases using gradient descent.
        """
        # target is a column vector
        target = target.reshape(-1, 1)
        T = len(self.xs)
        
        # gradient of the output layer
        # derivative of loss with softmax
        do = self.os[-1] - target
        dWy = np.dot(do, self.zs[-1].T)
        dby = do
        
        # backpropagate into the last hidden state
        dh = np.dot(self.Wy.T, do)
        
        # initialize gradients
        dWx = np.zeros_like(self.Wx)
        dWz = np.zeros_like(self.Wz)
        dbz = np.zeros_like(self.bz)
        
        # backpropagation through time
        for t in reversed(range(T)):
            # derivative of tanh: dtanh/dz = 1 - tanh(z)^2
            dz = dh * (1 - self.zs[t] * self.zs[t])
            dWx += np.dot(dz, self.xs[t].T)
            # for t=0, the previous hidden state is the initial state
            if t == 0:
                z_prev = np.zeros((self.hidden_size, 1))
            else:
                z_prev = self.zs[t - 1]
            dWz += np.dot(dz, z_prev.T)
            dbz += dz

            # propagate the gradient to the previous hidden state
            dh = np.dot(self.Wz.T, dz)
        
        # "Clip gradients" to avoid really large gradients 
        for grad in [dWx, dWz, dWy, dbz, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        # update parameters using gradient descent
        self.Wx -= self.learning_rate * dWx
        self.Wz -= self.learning_rate * dWz
        self.Wy -= self.learning_rate * dWy
        self.bz -= self.learning_rate * dbz
        self.by -= self.learning_rate * dby

    @staticmethod
    def softmax(x):
        """
        Compute softmax values for vector x in a numerically stable way.
        
        Args:
            x: Input vector (n x 1)
            
        Returns:
            Vector of same shape as x with softmax probabilities.
        """
        # softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

        x_shifted = x - np.max(x)
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x)

    def sample(self, z, seed_char, char_to_idx, idx_to_char, length=100, temperature=1.00):
        """
        Generate text starting with seed_char.
        
        Args:
            z: Initial hidden state
            seed_char: First character to start generation
            char_to_idx: Dictionary mapping characters to indices
            idx_to_char: Dictionary mapping indices to characters
            length: Number of characters to generate
            temperature: Controls randomness (lower = more conservative)
            
        Returns:
            str: Generated text of specified length.
        """
        x = np.zeros((len(char_to_idx), 1))
        x[char_to_idx[seed_char]] = 1
        generated = seed_char
        
        for _ in range(length):
            # Forward pass with single character
            o, z = self.forward([x], z)

            # Apply temperature scaling
            logits = np.log(o + 1e-10)
            exp_logits = np.exp(logits / temperature)
            probs = exp_logits / np.sum(exp_logits)

            # Sample next character from probability distribution
            idx = np.random.choice(len(probs), p=probs.ravel())
            next_char = idx_to_char[idx]
            generated += next_char
            x = np.zeros((len(char_to_idx), 1))
            x[char_to_idx[next_char]] = 1
        
        return generated

def create_char_mappings(text):
    """
    Create character-to-index and index-to-character mappings.
    
    Args:
        text: Input text
        
    Returns:
        tuple: (unique characters, char_to_idx dict, idx_to_char dict)
    """
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

def create_training_sequences(text, char_to_idx, seq_length=25):
    """
    Create training sequences and targets from input text.
    
    Args:
        text: Input text
        char_to_idx: Dictionary mapping characters to indices
        seq_length: Length of sequences to generate
        
    Returns:
        tuple: (X training sequences, y target characters)
    """
    sequences = []
    next_chars = []
    
    # Create sequences and their target next characters
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])
    
    # Convert to one-hot encoded vectors
    X = np.zeros((len(sequences), seq_length, len(char_to_idx)))
    y = np.zeros((len(sequences), len(char_to_idx)))
    
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1
    
    return X, y

def train_rnn_export(rnn, X, y, char_to_idx, idx_to_char, epochs=50, batch_size=32, print_every=10, export_file="results.txt"):
	"""
	Train the RNN and export training details to a log file.
	
	The log file will contain:
		- The hyperparameters used
		- CPU and power information
		- The training start time
		- Epoch number, average loss, and duration per epoch
		- A generated text sample at intervals defined by print_every
	
	Args:
		rnn: SimpleRNN instance.
		X: Training sequences.
		y: Target characters.
		char_to_idx: Character-to-index mapping.
		idx_to_char: Index-to-character mapping.
		epochs: Number of epochs to train.
		batch_size: Batch size for training.
		print_every: Interval (in epochs) at which to output a sample.
		export_file: Filename for exporting the log.
	"""
	n_samples = len(X)
	
	# Retrieve system information
	start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	cpu_info = platform.processor() or "Unknown CPU"
	machine_info = platform.machine() or "Unknown Machine"
	system_info = platform.system() or "Unknown System"
	
	try:
		battery = psutil.sensors_battery()
		if battery:
			battery_info = f"{battery.percent}% {'(Plugged in)' if battery.power_plugged else '(Not plugged in)'}"
		else:
			battery_info = "Battery info not available"
	except Exception as e:
		battery_info = f"Battery info retrieval error: {str(e)}"
	
	# Open log file for writing
	with open(export_file, "w", encoding="utf-8") as log_file:
		# Log hyperparameters and system information
		log_file.write("=== Training Log ===\n")
		log_file.write(f"Training start time: {start_time_str}\n")
		log_file.write(f"CPU: {cpu_info}, Machine: {machine_info}, System: {system_info}\n")
		log_file.write(f"Battery/Power Info: {battery_info}\n\n")
		
		log_file.write("Hyperparameters:\n")
		log_file.write(f"Sequence length: {X.shape[1]}\n")
		log_file.write(f"Input size (vocabulary size): {rnn.Wx.shape[1]}\n")
		log_file.write(f"Hidden size: {rnn.hidden_size}\n")
		log_file.write(f"Output size (vocabulary size): {rnn.Wy.shape[0]}\n")
		log_file.write(f"Learning rate: {rnn.learning_rate}\n")
		log_file.write(f"Epochs: {epochs}\n")
		log_file.write(f"Batch size: {batch_size}\n")
		log_file.write(f"Total training sequences: {n_samples}\n\n")
		
		log_file.write("Epoch Logs:\n")
		
		for epoch in range(epochs):
			epoch_start = time.time()
			total_loss = 0
			indices = np.random.permutation(n_samples)
			
			for start_idx in range(0, n_samples, batch_size):
				batch_indices = indices[start_idx:start_idx + batch_size]
				z = np.zeros((rnn.hidden_size, 1))
				batch_loss = 0
				
				for idx in batch_indices:
					inputs = X[idx]
					target = y[idx]
					
					# Forward pass
					o, z = rnn.forward(inputs, z)
					loss = -np.sum(target.reshape(-1, 1) * np.log(o + 1e-10))
					batch_loss += loss
					
					# Backward pass
					rnn.backward(target)
				
				total_loss += batch_loss
			
			avg_loss = total_loss / n_samples
			epoch_duration = time.time() - epoch_start
			current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
			epoch_log = f"Epoch {epoch}, Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f} sec, Timestamp: {current_time}\n"
			log_file.write(epoch_log)
			print(epoch_log.strip())
			
			if epoch % print_every == 0:
				# Generate a sample using the current state of the model
				z_sample = np.zeros((rnn.hidden_size, 1))
				sample_text = rnn.sample(z_sample, 'T', char_to_idx, idx_to_char, length=100)
				sample_log = f"Sample text at Epoch {epoch}:\n{sample_text}\n\n"
				log_file.write(sample_log)
				print(sample_log)
				
	print(f"Training log exported to {export_file}")
    
def load_data(filename):
    """
    Load text data from file with error handling.
    
    Args:
        filename: Path to text file
        
    Returns:
        str: Content of the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file can't be decoded as UTF-8
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text:
            raise ValueError("File is empty")
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"Could not decode file: {filename}. Please ensure it's a valid text file.")

# Test the data loading and processing pipeline
try:
    # Load the sonnets
    text = load_data('sonnets_sample.txt')
    print(f"Successfully loaded {len(text)} characters")
    
    # Create character mappings
    chars, char_to_idx, idx_to_char = create_char_mappings(text)
    print(f"Vocabulary size: {len(chars)} unique characters")
    
    # Create training sequences
    X, y = create_training_sequences(text, char_to_idx, seq_length=45)
    print(f"Created {len(X)} training sequences")
    
    # Print a sample sequence and its target
    sample_idx = 0
    sample_sequence = ''.join([idx_to_char[np.argmax(x)] for x in X[sample_idx]])
    sample_target = idx_to_char[np.argmax(y[sample_idx])]
    print(f"\nSample sequence: {sample_sequence}")
    print(f"Target character: {sample_target}")
    
    # creating and traing RNN model
    input_size = len(char_to_idx)
    hidden_size = 15
    output_size = len(char_to_idx)
    rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate=0.01)
    
    train_rnn_export(rnn, X, y, char_to_idx, idx_to_char, epochs=100, batch_size=4, print_every=1)
    
except Exception as e:
    print(f"Error processing data: {str(e)}")
