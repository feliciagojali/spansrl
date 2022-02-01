import numpy as np
def create_span(length, max_span_length):
    span_start = []
    span_end = []
    span_width = []
    for i in range(length):
        for j in range(max_span_length):
            start_idx = i
            end_idx = i+j

            if (end_idx >= length):
                break
            span_start.append(start_idx)
            span_end.append(end_idx)
            span_width.append(end_idx - start_idx + 1)

    # Shape span_start, span_end: [num_spans]
    return span_start, span_end, span_width

def split_into_batch(data, batch, filename):
    endpoint = len(data ) // batch
    start = 0
    for i in range(batch):
        if (i != batch-1):
            np.save(filename+str(i)+'.npy', data[start:start+endpoint])
        else:
            np.save(filename+str(i)+'.npy', data[start:])
        start += endpoint

def read_from_batch(filename, batch):
    data = []
    for i in range(batch):
        data.append(np.load(filename+str(i)+'.npy'))
    return np.concatenate(data)
