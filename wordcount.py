import os
import string 
import collections 
import itertools 
import time 
import operator 
import glob 
import multiprocessing
import sys

class SimpleMapReduce(object): 
  
    def __init__(self, map_func, reduce_func): 
        """ 
        map_func 
        Function to map inputs to intermediate data. Takes as 
        argument one input value and returns a tuple with the key 
        and a value to be reduced. 

        reduce_func 
        Function to reduce partitioned version of intermediate data 
        to final output. Takes as argument a key as produced by 
        map_func and a sequence of the values associated with that 
        key. 
        """ 
        self.map_func = map_func 
        self.reduce_func = reduce_func 

    def partition(self, mapped_values): 
        """Organize the mapped values by their key. 
        Returns an unsorted sequence of tuples with a key and a sequence of values.  """ 
        partitioned_data = collections.defaultdict(list) 
        for key, value in mapped_values: 
            partitioned_data[key].append(value) 
        return partitioned_data.items()

    def __call__(self, inputs, nprocs=multiprocessing.cpu_count()):
        """Process the inputs through the map and reduce functions given.

        Args:
            inputs: An iterable containing the input data to be processed.
            nprocs: The number of processes that we're going to use to process the data
        """ 
        # Divide each files into equal chunks for each process to process
        # Each file will be processed sequentially but multiple processes will process each file in parallel.
        file_chunk = [
            (filename, i, nprocs)
            for filename in inputs
            for i in range(nprocs)
        ]

        # Use a pool of workers to process the map and reduce functions in parallel
        with multiprocessing.Pool(processes=nprocs) as pool:
            map_responses = pool.starmap(self.map_func, file_chunk)
            flattened_responses = itertools.chain.from_iterable(counter.items() for counter in map_responses)
            partitioned_data = self.partition(flattened_responses)
    
        reduced_values = map(self.reduce_func, partitioned_data)
        return reduced_values 

 
def file_to_words(filename, idx, nprocs): 
    """
    Read a file and return a sequence of (word, occurances) values. 
    Args:
        filename: name of the file to be processed.
        idx: index of the process processing the file.
        nprocs: total number of processes.
    """ 
    # Pin worker process to a specific core
    start_t = time.perf_counter()
    STOP_WORDS = set([ 
    'a', 'an', 'and', 'are', 'as', 'be', 'by', 'for', 'if', 'in',  
    'is', 'it', 'of', 'or', 'py', 'rst', 'that', 'the', 'to', 'with',  ]) 
    TR = "".maketrans(string.punctuation, ' ' * len(string.punctuation)) 
    
    file_size = os.path.getsize(filename)

    # Compute start and end byte positions for this thread
    start = idx * (file_size // nprocs)
    end = (idx + 1) * (file_size // nprocs) if idx < nprocs - 1 else file_size   

    output = collections.Counter()
    with open(filename, 'rt', errors='replace') as f:
        # Move to the start position
        f.seek(start)
        # Skip partial lines, we'll start at the beginning of the next line instead
        if start != 0:
            f.readline()
        pos = f.tell()

        # Read lines until reaching the end position
        while pos < end:
            line = f.readline()
            if not line:
                break

            # Append word and count
            if line.lstrip().startswith('..'): # Skip rst comment lines
                continue
            line = line.translate(TR) # Strip punctuation
            for word in line.split():
                word = word.lower()
                if word.isalpha() and word not in STOP_WORDS:
                    # KV pair will now be (word, count) instead of (word, 1)
                    output[word] += 1
            
            # Update the current position
            pos = f.tell()

    end_t = time.perf_counter()
    print(f'Process {idx} finished processing {filename} in {end_t - start_t} seconds')
    return output

def count_words(item): 
    """Convert the partitioned data for a word to a 
    tuple containing the word and the number of occurances. 
    """ 
    word, occurances = item 
    return (word, sum(occurances)) 

if __name__ == '__main__': 
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        nprocs = int(sys.argv[2])
        if nprocs < 1:
            nprocs = multiprocessing.cpu_count()
    else:
        print("Usage: python wordcount.py <relative_input_path> <num_processes>")
        sys.exit(1)

    start_time = time.time()  
    input_files = glob.glob(f'{file_path}/*') 

    mapper = SimpleMapReduce(file_to_words, count_words) 
    word_counts = mapper(input_files, nprocs) 
    #print(list(word_counts)) 
    word_counts = sorted(word_counts, key=operator.itemgetter(1)) 
    word_counts.reverse() 

    print('\nTOP 20 WORDS BY FREQUENCY\n') 
    top20 = word_counts[0:20] 
    longest = max(len(word) for word, count in top20) 
    i = 1 
    for word, count in top20: 
        print('%s.\t%-*s: %5s' % (i, longest+1, word, count)) 
        i = i + 1 

    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print("Elapsed Time: {} seconds".format(elapsed_time)) 
