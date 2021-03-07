from _utils import *


class VariationalAutoencoder(nn.Module):
    def __init__(self, teacher_forcing, eps_i, dropout, nbr_of_pitches, device,
                 input_size, enc_hidden_size, latent_features, dec_hidden_size):
        super(VariationalAutoencoder, self).__init__()

        self.device = device
        self.teacher_forcing = teacher_forcing
        self.eps_i = eps_i
        self.dropout = dropout
        self.nbr_of_pitches = nbr_of_pitches
        self.input_size = input_size
        self.enc_hidden_size = enc_hidden_size
        self.latent_features = latent_features
        self.dec_hidden_size = dec_hidden_size

        self.encoder = torch.nn.LSTM(batch_first=True, input_size=self.input_size, hidden_size=self.enc_hidden_size,
                                     num_layers=1, bidirectional=True)  # data goes into bidirectional encoder
        # encoded into a linear layer. inputs must be*2 because LSTM bidirectional, output must be 2*latent_space
        # because it needs split into mu and sigma right after.
        self.encoderOut = nn.Linear(in_features=self.enc_hidden_size * 2, out_features=self.latent_features * 2)
        # after being converted data goes through a fully connected layer
        self.linear_z = nn.Linear(in_features=self.latent_features, out_features=dec_hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.worddropout = nn.Dropout2d(self.dropout)
        # Define the conductor and note decoder
        self.conductor = nn.LSTM(dec_hidden_size, dec_hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(self.nbr_of_pitches + dec_hidden_size, dec_hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(dec_hidden_size, self.nbr_of_pitches)  # Linear note to note type (classes/pitches)

    # used to initialize the hidden layer of the encoder to zero before every batch
    def init_hidden_to_null(self, batch_size):
        # must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, self.enc_hidden_size, device=self.device)
        c0 = torch.zeros(2, batch_size, self.enc_hidden_size, device=self.device)
        # 2 because has 2 layers, n_layers_conductor
        init_conductor = torch.zeros(1, batch_size, self.dec_hidden_size, device=self.device)
        c_condunctor = torch.zeros(1, batch_size, self.dec_hidden_size, device=self.device)
        return init, c0, init_conductor, c_condunctor

    # Coin toss to determine whether to use teacher forcing on a note(Scheduled sampling)
    # Will always be True for eps_i = 1.
    def use_teacher_forcing(self):
        with torch.no_grad():
            tf = np.random.rand(1)[0] <= self.eps_i
        return tf

    def set_scheduled_sampling(self, eps_i):
        self.eps_i = eps_i

    def forward(self, x):
        batch_size = x.size(0)
        note = torch.zeros(batch_size, 1, self.nbr_of_pitches, device=self.device)
        the_input = torch.cat([note, x], dim=1)
        outputs = {}
        # creates hidden layer values
        h0, c0, hconductor, cconductor = self.init_hidden(batch_size)
        x = self.worddropout(x)
        # resets encoder at the beginning of every batch and gives it x
        x, hidden = self.encoder(x, (h0, c0))
        # x=self.dropout(x)
        # goes from 4096 to 1024
        x = self.encoderOut(x)
        # x=self.dropout(x)
        # Split encoder outputs into a mean and variance vector
        mu, log_var = torch.chunk(x, 2, dim=-1)
        # Make sure that the log variance is positive
        log_var = softplus(log_var)

        # :- Reparametrisation trick
        # a sample from N(mu, sigma) is mu + sigma * epsilon
        # where epsilon ~ N(0, 1)

        # Don't propagate gradients through randomness
        with torch.no_grad():
            batch_size = mu.size(0)
            epsilon = torch.randn(batch_size, 1, self.latent_features)

            if self.device == 'cuda':
                epsilon = epsilon.cuda()

        # setting sigma
        sigma = torch.exp(log_var * 2)
        # generate z - latent space
        z = mu + epsilon * sigma

        # decrese space
        z = self.linear_z(z)
        # z=self.dropout(z)
        # make dimensions fit (NOT SURE IF THIS IS ENTIRELY CORRECT)
        # z = z.permute(1,0,2)

        # DECODER ##############
        conductor_hidden = (hconductor, cconductor)
        counter = 0
        NOTESPERBAR = 16  # total notes in one bar
        totalbars = 16  # total bars as input
        TOTAL_NOTES = NOTESPERBAR*totalbars
        notes = torch.zeros(batch_size, TOTAL_NOTES, self.nbr_of_pitches, device=self.device)
        # For the first timestep the note is the embedding
        note = torch.zeros(batch_size, 1, self.nbr_of_pitches, device=self.device)

        # Go through each element in the latent sequence
        for i in range(16):
            embedding, conductor_hidden = self.conductor(z[:, i, :].view(batch_size, 1, -1), conductor_hidden)

            if self.use_teacher_forcing():
                # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(1, batch_size, self.dec_hidden_size, device=self.device),
                                  torch.randn(1, batch_size, self.dec_hidden_size, device=self.device))
                embedding = embedding.expand(batch_size, NOTESPERBAR, embedding.shape[2])
                e = torch.cat([embedding, the_input[:, range(i * 16, i * 16 + 16), :]], dim=-1)
                notes2, decoder_hidden = self.decoder(e, decoder_hidden)
                aux = self.linear(notes2)
                aux = torch.softmax(aux, dim=2);
                # generates 16 notes per batch at a time
                notes[:, range(i * 16, i * 16 + 16), :] = aux;
            else:
                # Reset the decoder state of each 16 bar sequence
                decoder_hidden = (torch.randn(1, batch_size, self.dec_hidden_size, device=self.device),
                                  torch.randn(1, batch_size, self.dec_hidden_size, device=self.device))
                sequence_length = 16  # notes per decoder
                for _ in range(sequence_length):
                    # Concat embedding with previous note
                    e = torch.cat([embedding, note], dim=-1)
                    e = e.view(batch_size, 1, -1)
                    # Generate a single note (for each batch)
                    note, decoder_hidden = self.decoder(e, decoder_hidden)
                    aux = self.linear(note)
                    aux = torch.softmax(aux, dim=2);
                    notes[:, counter, :] = aux.squeeze();
                    note = aux
                    counter = counter + 1

        outputs["x_hat"] = notes
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["log_var"] = log_var
        return outputs
