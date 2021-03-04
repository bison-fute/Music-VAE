from _utils import *
from IPython.display import clear_output

data_path = "data"  # path to data we want to use
midi_files = glob.glob(os.path.join(data_path, "**/*.mid"), recursive=True)  # list all mid files in data
nb_of_midi = len(midi_files)

saved_columns = [pretty_midi.note_number_to_name(n) for n in range(48,108)]
piano_rolls = pd.DataFrame(columns=['piano_roll_name', 'timestep'] + saved_columns)
piano_rolls = piano_rolls.set_index(['piano_roll_name', 'timestep'])


class Preprocessing:
    def __init__(self, midi_files):
        self.midi_files = midi_files
        self.nb_of_midi = len(midi_files)

        self.start = 0
        self.limit = -1
        self.file_name = "piano_rolls.csv"

    def list_of_all_the_instruments(self, limit=-1):
        _instrument_array = [[]]
        _subset_of_midi_files = self.midi_files[:limit]  # keep only the limit-th first files
        _nb_of_midi_in_subset = len(_subset_of_midi_files)
        for idx, file in enumerate(_subset_of_midi_files):
            _filename = os.path.basename(file)
            clear_output(wait=True)
            print("file {}/{}: Loading and parsing {}".format(idx, _nb_of_midi_in_subset, _filename))
            try:
                _pm_file = pretty_midi.PrettyMIDI(file)  # create a PrettyMidi object
                _instruments = _pm_file.instruments
                for instrument in _instruments:
                    _instrument_array.append([instrument.program, file, instrument.name.replace(';', '')])
            except:
                continue  # ignore unreadable files

        _instrument_df = pd.DataFrame(_instrument_array,
                                      columns=["instrument_track_nb", "filepath", "file_instrument_name"])
        _instrument_df.dropna(inplace=True)
        self.instrument_df = _instrument_df
        return self.instrument_df

    def most_common_instruments(self):
        _instrument_df = self.instrument_df.drop(columns=["file_instrument_name"])  # not needed in the output df
        _frequent_tracks_df = _instrument_df.groupby("instrument_track_nb").nunique()
        _frequent_tracks_df.rename(columns={"filepath": "nb_of_track_using_it"}, inplace=True)
        _frequent_tracks_df.sort_values("nb_of_track_using_it", ascending=False, inplace=True)
        _track_names_array = []
        for track_nb in _frequent_tracks_df.index.values:
            _track_name = pretty_midi.program_to_instrument_name((int(track_nb)))
            _track_names_array.append(_track_name)
        _frequent_tracks_df["midi_associated_instrument"] = _track_names_array
        self.frequent_tracks_df = _frequent_tracks_df
        return self.frequent_tracks_df

    def keep_only_piano_tracks(self, show_justification=False):
        """
        :return: couple (Dataframe, int) with a Dataframe of similar shape as instrument_df, making the link between the
        filepath and the associated instrument_track_nb_field
        """
        _five_most_used_tracks_nbrs = int(self.frequent_tracks_df.index[0])  # [int(self.frequent_tracks_df.index[i]) for i in range(int(len(self.frequent_tracks_df)/20))]
        _piano_instruments_df = self.instrument_df[
            ((self.instrument_df["instrument_track_nb"] == _five_most_used_tracks_nbrs)) &
            (self.instrument_df["file_instrument_name"].str.contains("piano", case=False))]
        if show_justification:
            pass
        self._piano_instruments_df = _piano_instruments_df
        return _piano_instruments_df, len(_piano_instruments_df)

    # The list of track having a piano melody is built. The next objective is to iterate on it to load the MIDI files,
    # get the piano rolls and convert them in a CSV

    def write_one_instrument_into_csv(self, instrument, _song_name, _semi_shift, _sampling_freq, i, j):
        # Hardcoded for now:
        if (
                instrument.program == 0 | instrument.program == 1 | instrument.program == 2 | instrument.program == 3 | instrument.program == 4) \
                and 'piano' in instrument.name.lower():
            """ Generate a unique top level index per song and instrument in this song, 
            if it has multiples of the same kind. """
            for note in instrument.notes:
                note.pitch += _semi_shift
            try:
                _df = encode_dummies(instrument, _sampling_freq).fillna(value=0)  # fill invalid values
            except Exception as e:  # the piano roll is bad, don't use this track
                print("Encountered exception for song {}, instrument {}: {}".format(
                    _song_name, instrument.name, e))
            print("One hot encoded into piano rolls")
            # chop before doing anything else to conserve memory
            _df = chopster(_df)
            print("Chopped to relevant notes only")
            _df = trim_blanks(_df)
            print("Fast forwarded to first note playing")
            if _df is None:
                return None
            _df = minister(_df)
            print("Reduced velocity to on/off")
            _df = arpster(_df)
            print("Chords to arpegs")
            # ensures that files with more than 1 note pr timestep is not added to the dataset.
            if np.amax(np.asarray(_df.astype(bool).sum(axis=1))) > 1:
                return None
            _df.reset_index(inplace=True, drop=True)
            top_level_index = "{}_{}:{}".format(_song_name, i, j)
            _df['timestep'] = _df.index
            _df['piano_roll_name'] = top_level_index
            _df = _df.set_index(['piano_roll_name', 'timestep'])
            _df.to_csv(self.file_name, sep=';', mode='a', encoding='utf-8', header=False)

    def write_one_track_into_csv(self, i, file):
        _song_name = os.path.basename(file)
        clear_output(wait=True)
        print("{}/{}: {}.".format(i, len(self._piano_instruments_df), _song_name))
        try:
            _semi_shift = transposer(file)
            pm = pretty_midi.PrettyMIDI(file)
            print("Loaded into memory, processing...")
            """ Here we calculate the amount of seconds per sixteenth note, by taking the second beat of the song 
                (which is the same as the difference in seconds between the first and second beat), and convert it 
                to the sampling frequency format that pretty_midi expects """
            _sampling_freq = 1 / (pm.get_beats()[1] / 4)
        except Exception as e:
            print("Ignoring song {}: {}".format(_song_name, e))  # ignore files we can't load
            return None
        for j, instrument in enumerate(pm.instruments):
            self.write_one_instrument_into_csv(instrument, _song_name, _semi_shift, _sampling_freq, i, j)

    def write_all_tracks_into_csv(self):
        for i, file in enumerate(self._piano_instruments_df['filepath'][self.start:self.limit]):
            clear()
            self.write_one_track_into_csv(i, file)








