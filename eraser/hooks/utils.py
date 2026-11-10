"""Map unpadded prompt indices to left-padded model inputs."""


def map_prompt_token_indices_with_left_padding(indexes, attention_mask, sequence_length):
    valid_length = int(attention_mask.sum().item())
    padding = len(attention_mask) - valid_length
    return sorted({int(index) + padding for index in indexes
                   if index is not None and 0 <= int(index) < valid_length
                   and int(index) + padding < sequence_length})
