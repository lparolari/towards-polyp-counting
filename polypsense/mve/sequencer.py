SequenceId = str
ItemId = int
Sequence = list[ItemId]


class Sequencer:
    """
    Build sequences out of a collection of items based on a `sequence_key` in
    the item's annotation. If the `sequence_key` is not present, all items
    belongs to the `default` sequence.
    """

    def __init__(self, ds, sequence_key: str = "sequence_id"):
        self.ds = ds
        self.sequence_key = sequence_key
        self._build_sequence_index()

    def list_sequence_ids(self) -> list[SequenceId]:
        """
        Returns a list of all sequence ids.
        """
        return list(self._sequence2index.keys())

    def get_sequence(self, sequence_id: SequenceId) -> Sequence:
        """
        Returns a list of item ids that belong to the given sequence.
        """
        return sorted(self._sequence2index[sequence_id])

    def find_sequence_id(self, item_id: ItemId) -> SequenceId:
        """
        Returns the sequence id for the given item id.
        """
        return self._index2sequence[item_id]

    def _build_sequence_index(self):
        """
        Builds two indices:
        * sequence -> list of indices, with the list of indices belonging to given sequence
        * index -> sequence, with the sequence id for each given index
        """
        sequence2index = {}
        index2sequence = {}

        for i in range(len(self.ds)):
            id = self.ds.id(i)
            tgt_ann = self.ds.tgt_ann(id)

            sequence_id = tgt_ann.get(self.sequence_key, "default")

            if sequence_id not in sequence2index:
                sequence2index[sequence_id] = []

            sequence2index[sequence_id].append(id)
            index2sequence[id] = sequence_id

        assert len(index2sequence) == len(
            [v for vs in sequence2index.values() for v in vs]
        )

        self._sequence2index = sequence2index
        self._index2sequence = index2sequence
