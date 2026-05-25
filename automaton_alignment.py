"""
Automaton Alignment Module

This module provides functionality to convert character-level finite state automata (FSA)
to token-level automata compatible with subword tokenizers like BERT's WordPiece tokenizer.

Dependencies:
    pip install interegular transformers torch

The implementation is based on the algorithm described in:
"Efficient Guided Generation for Large Language Models" by Willard & Louf (2023)
https://arxiv.org/abs/2307.09702

Compatible with HuggingFace tokenizers, particularly:
- BertTokenizer (used in RDLM for lm1b dataset)
- GPT2Tokenizer
- Other AutoTokenizer instances
"""

import re
from typing import Dict, Set, Tuple, List, Optional, Union
from dataclasses import dataclass
import interegular
from interegular.fsm import FSM


class SimpleTokenizerWrapper:
    """
    Wrapper to make a simple dict tokenizer compatible with HuggingFace tokenizer interface.

    This is useful for Diffusion-LM models that use simple dict tokenizers like:
        {token_id: token_string}

    The wrapper provides the minimal interface needed by AutomatonAligner:
    - get_vocab()
    - decode()
    - convert_ids_to_tokens()

    Example:
        >>> tokenizer_dict = {0: 'hello', 1: 'world', 2: '!'}
        >>> wrapped = SimpleTokenizerWrapper(tokenizer_dict)
        >>> wrapped.get_vocab()
        {'hello': 0, 'world': 1, '!': 2}
    """

    def __init__(self, tokenizer_dict: Dict[int, str]):
        """
        Initialize wrapper with a dict tokenizer.

        Args:
            tokenizer_dict: Dictionary mapping token_id (int) -> token_string (str)
        """
        self.id_to_token_dict = tokenizer_dict
        # Reverse to get token_string -> token_id (HuggingFace format)
        self.token_to_id_dict = {v: k for k, v in tokenizer_dict.items()}
        self.vocab_size = len(tokenizer_dict)

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary as dict mapping token_string -> token_id."""
        return self.token_to_id_dict

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = False) -> str:
        """
        Decode a list of token IDs to a string.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Ignored (for compatibility)
            clean_up_tokenization_spaces: Ignored (for compatibility)

        Returns:
            Decoded string (tokens joined with spaces)
        """
        tokens = [self.id_to_token_dict.get(tid, '<UNK>') for tid in token_ids]
        return ' '.join(tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        """
        Convert a single token ID to its token string.

        Args:
            token_id: Token ID to convert

        Returns:
            Token string
        """
        return self.id_to_token_dict.get(token_id, '<UNK>')

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size


@dataclass
class TokenAutomaton:
    """
    Represents a token-level automaton aligned with a subword tokenizer.

    Attributes:
        states: Set of all states in the automaton
        initial_state: The starting state
        final_states: Set of accepting/final states
        transitions: Mapping from (state, token_id) -> next_state
        state_to_tokens: Mapping from state -> set of valid token_ids
        alphabet_size: Number of tokens in the vocabulary
        state_list: Ordered list of states (sorted for consistency)
        state_to_idx: Mapping from state to its index in state_list
        state_transitions: Pre-computed transition data for efficient score computation
                          List where state_transitions[state_idx] = (valid_token_ids, dest_state_indices)
                          Both as Python lists that can be converted to tensors on appropriate device
    """
    states: Set[int]
    initial_state: int
    final_states: Set[int]
    transitions: Dict[Tuple[int, int], int]
    state_to_tokens: Dict[int, Set[int]]
    alphabet_size: int
    state_list: List[int]
    state_to_idx: Dict[int, int]
    state_transitions: List[Tuple[List[int], List[int]]]

    def get_valid_tokens(self, state: int) -> Set[int]:
        """Get all valid token IDs from a given state."""
        return self.state_to_tokens.get(state, set())

    def transition(self, state: int, token_id: int) -> Optional[int]:
        """Get the next state given current state and token_id."""
        return self.transitions.get((state, token_id))

    def get_transitions(self, state: int) -> Dict[int, int]:
        """
        Get all transitions from a given state.

        Args:
            state: The state to get transitions from

        Returns:
            Dictionary mapping token_id -> next_state for all valid transitions
        """
        result = {}
        for token_id in self.get_valid_tokens(state):
            next_state = self.transition(state, token_id)
            if next_state is not None:
                result[token_id] = next_state
        return result

    def is_final(self, state: int) -> bool:
        """Check if a state is a final/accepting state."""
        return state in self.final_states

    def __repr__(self):
        return (f"TokenAutomaton(states={len(self.states)}, "
                f"transitions={len(self.transitions)}, "
                f"vocab_size={self.alphabet_size})")


class AutomatonAligner:
    """
    Aligns character-level FSAs with subword tokenizers.

    This class converts regular expressions (or FSMs) defined over characters
    into token-level automata that work with tokenizers like BERT's WordPiece.
    """

    def __init__(self, tokenizer):
        """
        Initialize the aligner with a tokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer instance (e.g., BertTokenizer)
        """
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)

        # Create reverse mapping: token_id -> token_str
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def regex_to_fsm(self, regex_pattern: str) -> FSM:
        """
        Convert a Python regex pattern to an interegular FSM.

        Args:
            regex_pattern: A Python regular expression string

        Returns:
            An interegular FSM object representing the regex
        """
        # interegular doesn't support $ anchors. Escape any bare $ (not
        # preceded by \) so that literal dollar signs (e.g. "$schema") don't
        # crash the parser.
        regex_pattern = re.sub(r'(?<!\\)\$', r'\\$', regex_pattern)

        # Parse the regex pattern using interegular
        parsed = interegular.parse_pattern(regex_pattern)

        # Convert to FSM and make it deterministic
        fsm = parsed.to_fsm()

        # Reduce the FSM (minimize it)
        fsm = fsm.reduce()

        return fsm

    def _walk_fsm(self, fsm: FSM, from_state: int, input_string: str) -> Optional[int]:
        """
        Walk the character-level FSM with an input string from a given state.

        Args:
            fsm: The character-level FSM
            from_state: Starting state
            input_string: String to process

        Returns:
            The ending state if successful, None if the walk fails
        """
        current_state = from_state
        alphabet = fsm.alphabet

        for char in input_string:
            # Map character to symbol using the FSM's alphabet
            symbol = alphabet[char]

            # Check if there's a transition for this symbol
            if current_state in fsm.map and symbol in fsm.map[current_state]:
                current_state = fsm.map[current_state][symbol]
            else:
                return None

        return current_state

    def _decode_token(self, token_id: int) -> str:
        """
        Decode a token ID to its string representation.

        Args:
            token_id: The token ID to decode

        Returns:
            The decoded string
        """
        try:
            # Use tokenizer.decode for proper handling of special tokens
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return decoded
        except Exception:
            # Fallback: use convert_ids_to_tokens
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token_str, str):
                # For WordPiece, remove ## prefix
                if token_str.startswith('##'):
                    return token_str[2:]
                return token_str
            return str(token_str)

    def create_token_automaton(
        self,
        regex_pattern: Optional[str] = None,
        fsm: Optional[FSM] = None,
        add_terminal_states: bool = True
    ) -> TokenAutomaton:
        """
        Create a token-level automaton from a regex or FSM.

        Args:
            regex_pattern: A Python regex pattern (if fsm is not provided)
            fsm: An interegular FSM object (if regex_pattern is not provided)
            add_terminal_states: If True, add terminal states for END token handling
                                (for Diffusion-LM e2e dataset). Set to False for
                                tokenizers without explicit END tokens (like PLAID).

        Returns:
            A TokenAutomaton object with token-level transitions

        Raises:
            ValueError: If neither regex_pattern nor fsm is provided
        """
        if fsm is None:
            if regex_pattern is None:
                raise ValueError("Either regex_pattern or fsm must be provided")
            fsm = self.regex_to_fsm(regex_pattern)

        # Initialize the token automaton structure
        states = set(fsm.states)
        initial_state = fsm.initial
        final_states = set(fsm.finals)
        transitions: Dict[Tuple[int, int], int] = {}
        state_to_tokens: Dict[int, Set[int]] = {state: set() for state in states}

        # For each token in the vocabulary
        for token_id in range(self.vocab_size):
            # Decode the token to get its character sequence
            token_str = self._decode_token(token_id)

            # For each state in the FSM, try to walk with this token
            for state in states:
                end_state = self._walk_fsm(fsm, state, token_str)

                if end_state is not None:
                    # This token is valid from this state
                    transitions[(state, token_id)] = end_state
                    state_to_tokens[state].add(token_id)

        # Add terminal states for END token handling (optional)
        # This prevents tokens after END from affecting the automaton score
        # Disable for tokenizers without explicit END tokens (like PLAID)
        end_token_id = None
        if add_terminal_states:
            for tid, tstr in self.id_to_token.items():
                if tstr.strip() == 'END':
                    end_token_id = tid
                    break

        if end_token_id is not None and add_terminal_states:
            # Create two new terminal states
            # Use IDs that don't conflict with FSM state IDs
            max_state = max(states)
            terminal_accept = max_state + 1
            terminal_reject = max_state + 2

            # Save original states before modification
            original_states = set(states)
            original_final_states = set(final_states)

            # Add terminal states
            states.add(terminal_accept)
            states.add(terminal_reject)

            # IMPORTANT: Make ONLY terminal states final
            # This effectively changes the regex from .*Chinese.*food.* to .*Chinese.*food.*END
            # Sequences without END will get low scores, forcing the model to generate END
            final_states.clear()  # Remove original final states (e.g., State 11)
            final_states.add(terminal_accept)  # Only Terminal_Accept is final

            # Initialize state_to_tokens for terminal states
            state_to_tokens[terminal_accept] = set()
            state_to_tokens[terminal_reject] = set()

            # Add END transitions from all original states
            for state in original_states:
                if state in original_final_states:
                    # Final state + END → Terminal_Accept
                    transitions[(state, end_token_id)] = terminal_accept
                    state_to_tokens[state].add(end_token_id)
                else:
                    # Non-final state + END → Terminal_Reject
                    transitions[(state, end_token_id)] = terminal_reject
                    state_to_tokens[state].add(end_token_id)

            # Add self-loops for terminal states on ALL tokens
            for token_id in range(self.vocab_size):
                # Terminal_Accept loops to itself
                transitions[(terminal_accept, token_id)] = terminal_accept
                state_to_tokens[terminal_accept].add(token_id)

                # Terminal_Reject loops to itself
                transitions[(terminal_reject, token_id)] = terminal_reject
                state_to_tokens[terminal_reject].add(token_id)

        # Create consistent ordering of states
        state_list = sorted(states)
        state_to_idx = {state: idx for idx, state in enumerate(state_list)}

        # Pre-compute transition data for efficient score computation
        # For each state, store (list of valid token IDs, list of destination state indices)
        state_transitions = []
        for state in state_list:
            valid_tokens = list(state_to_tokens[state])
            if len(valid_tokens) == 0:
                # No valid transitions from this state
                state_transitions.append(([], []))
            else:
                # Compute destination state index for each valid token
                dest_indices = [
                    state_to_idx[transitions[(state, token_id)]]
                    for token_id in valid_tokens
                ]
                state_transitions.append((valid_tokens, dest_indices))

        return TokenAutomaton(
            states=states,
            initial_state=initial_state,
            final_states=final_states,
            transitions=transitions,
            state_to_tokens=state_to_tokens,
            alphabet_size=self.vocab_size,
            state_list=state_list,
            state_to_idx=state_to_idx,
            state_transitions=state_transitions
        )

    def align_automaton(
        self,
        regex_pattern: str,
        return_index: bool = False
    ) -> Union[TokenAutomaton, Dict[int, Dict[int, int]]]:
        """
        Align a regex pattern with the tokenizer vocabulary.

        This is the main entry point for the alignment process.

        Args:
            regex_pattern: A Python regular expression string
            return_index: If True, return the index dict instead of TokenAutomaton

        Returns:
            Either a TokenAutomaton object or an index dictionary
            mapping states to {token_id -> next_state}
        """
        token_automaton = self.create_token_automaton(regex_pattern=regex_pattern)

        if return_index:
            # Convert to the index format (state -> {token_id -> next_state})
            index = {}
            for state in token_automaton.states:
                index[state] = {}
                for token_id in token_automaton.get_valid_tokens(state):
                    next_state = token_automaton.transition(state, token_id)
                    if next_state is not None:
                        index[state][token_id] = next_state
            return index

        return token_automaton


def load_bert_tokenizer(model_name: str = 'bert-base-uncased'):
    """
    Convenience function to load a BERT tokenizer.

    Args:
        model_name: Name of the pretrained BERT model

    Returns:
        A HuggingFace BertTokenizer instance
    """
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(model_name)


def example_usage():
    """
    Example usage of the automaton alignment functionality.
    """
    from transformers import BertTokenizer

    # Load BERT tokenizer (same as used in RDLM for lm1b dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)

    # Create aligner
    aligner = AutomatonAligner(tokenizer)

    # Define a simple regex pattern
    # Example: match "cat" or "dog"
    regex_pattern = r"(cat|dog)"

    # Create token-level automaton
    token_automaton = aligner.align_automaton(regex_pattern)

    print(f"Created {token_automaton}")
    print(f"Initial state: {token_automaton.initial_state}")
    print(f"Final states: {token_automaton.final_states}")

    # Check what tokens are valid from the initial state
    valid_tokens = token_automaton.get_valid_tokens(token_automaton.initial_state)
    print(f"\nValid tokens from initial state: {len(valid_tokens)}")

    # Show some example valid tokens
    for token_id in list(valid_tokens)[:10]:
        token_str = tokenizer.decode([token_id])
        print(f"  Token {token_id}: '{token_str}'")

    # Simulate walking the automaton with the token sequence for "cat"
    cat_tokens = tokenizer.encode("cat", add_special_tokens=False)
    print(f"\nTokens for 'cat': {cat_tokens}")

    current_state = token_automaton.initial_state
    for token_id in cat_tokens:
        next_state = token_automaton.transition(current_state, token_id)
        token_str = tokenizer.decode([token_id])
        print(f"  State {current_state} --[{token_id}:'{token_str}']-> {next_state}")
        if next_state is None:
            print("  Failed: invalid transition")
            break
        current_state = next_state

    if current_state is not None:
        print(f"  Final state {current_state} is accepting: {token_automaton.is_final(current_state)}")


if __name__ == "__main__":
    import sys

    # Only run example if explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_usage()
    else:
        print("Automaton Alignment Module")
        print("=" * 60)
        print("Import this module to use it:")
        print("  from automaton_alignment import AutomatonAligner")
        print("\nTo run the example:")
        print("  python automaton_alignment.py --example")
        print("\nNote: Example requires downloading BERT tokenizer.")
        print("Use with your existing tokenizer instead.")
        print("=" * 60)
