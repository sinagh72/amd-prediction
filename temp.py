# if __name__ == "__main__":
# parser = argparse.ArgumentParser()
# parser.parse_args()
# parser.add_argument("-d", "--d_model", default=512, type=int,
#                     help="the number of expected features in the encoder/decoder inputs")
#
# parser.add_argument("-nh", "--n_head", default=8, type=int,
#                     help="the number of heads in the multiheadattention models (default=8)")
#
# parser.add_argument("-ne", "--num_encoder_layers", default=6, type=int,
#                     help="the number of sub-encoder-layers in the encoder (default=6)")
#
# parser.add_argument("-nd", "--num_decoder_layers", default=6, type=int,
#                     help="the number of sub-decoder-layers in the decoder (default=6)")
#
# parser.add_argument("-f", "--dim_feedforward", default=2048, type=int,
#                     help="the dimension of the feedforward network model (default=2048)")
#
# parser.add_argument("-dr", "--dropout", default=0.1, type=float,
#                     help="the dropout value (default=0.1)")
#
# parser.add_argument("-a", "--activation", default="relu", type=str, choices=["relu", "gelu"],
#                     help="the activation function of encoder/decoder intermediate layer, "
#                          "can be a string (“relu” or “gelu”) or a unary callable. Default: relu")
#
# parser.add_argument("-le", "--layer_norm_eps", default=1e-5, type=float,
#                     help="the eps value in layer normalization components (default=1e-5)")
#
# parser.add_argument("-b", "--batch_first", action="store_true",
#                     help="If True, then the input and output tensors are provided as (batch, seq, feature). "
#                          "Default: False (seq, batch, feature)")
#
# parser.add_argument("-nf", "--norm_first", action="store_true",
#                     help="if True, encoder and decoder layers will perform LayerNorms before other attention and "
#                          "feedforward operations, otherwise after. Default: False (after)")
#
# args = parser.parse_args()

# d_model = args.d_model
# nhead = args.nhead
# num_encoder_layers = args.num_encoder_layers
# num_decoder_layers = args.num_decoder_layers
# dim_feedforward = args.dim_feedforward
# dropout = args.dropout
# activation = args.activation
# custom_encoder = args.custom_encoder
# custom_decoder = args.custom_decoder
# layer_norm_eps = args.layer_norm_eps
# batch_first = args.batch_first
# norm_first = args.norm_first

# tf_model = torch.nn.Transformer(d_model=d_model,
#                                 nhead=nhead,
#                                 num_encoder_layers=num_encoder_layers,
#                                 num_decoder_layers=num_decoder_layers,
#                                 dim_feedforward=dim_feedforward,
#                                 dropout=dropout,
#                                 activation=activation,
#                                 custom_encoder=custom_encoder,
#                                 custom_decoder=custom_decoder,
#                                 layer_norm_eps=layer_norm_eps,
#                                 batch_first=batch_first,
#                                 norm_first=norm_first)


# X_train = torch.tensor(X_train)
# print(X_train.shape)
