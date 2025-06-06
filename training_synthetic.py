{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red131\green0\blue165;\red245\green245\blue245;\red0\green0\blue0;
\red15\green112\blue1;\red19\green85\blue52;\red0\green0\blue255;\red86\green65\blue25;\red31\green99\blue128;
\red144\green1\blue18;}
{\*\expandedcolortbl;;\cssrgb\c59216\c13725\c70588;\cssrgb\c96863\c96863\c96863;\cssrgb\c0\c0\c0;
\cssrgb\c0\c50196\c0;\cssrgb\c6667\c40000\c26667;\cssrgb\c0\c0\c100000;\cssrgb\c41569\c32157\c12941;\cssrgb\c14510\c46275\c57647;
\cssrgb\c63922\c8235\c8235;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import\cf0 \strokec4  torch\cb1 \
\cf2 \cb3 \strokec2 from\cf0 \strokec4  torch.optim \cf2 \strokec2 import\cf0 \strokec4  AdamW\cb1 \
\cf2 \cb3 \strokec2 from\cf0 \strokec4  torch.nn \cf2 \strokec2 import\cf0 \strokec4  BCEWithLogitsLoss\cb1 \
\
\cf5 \cb3 \strokec5 # === Define group representations (replace with actual zn_irreps and zn_transitions) ===\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 # Assume these functions return reps and transitions for \uc0\u8484 _10\cf0 \cb1 \strokec4 \
\cb3 G, reps = zn_irreps(\cf6 \strokec6 10\cf0 \strokec4 )\cb1 \
\cb3 transitions = zn_transitions(\cf6 \strokec6 10\cf0 \strokec4 )\cb1 \
\
\cf5 \cb3 \strokec5 # === Model Setup ===\cf0 \cb1 \strokec4 \
\cb3 model = ScalableAlgebraicTransformer(reps, transitions, depth=\cf6 \strokec6 3\cf0 \strokec4 )\cb1 \
\cb3 optimizer = AdamW(model.parameters(), lr=\cf6 \strokec6 3e-4\cf0 \strokec4 , weight_decay=\cf6 \strokec6 1e-5\cf0 \strokec4 )\cb1 \
\cb3 criterion = BCEWithLogitsLoss()\cb1 \
\
\cf5 \cb3 \strokec5 # === Training Loop ===\cf0 \cb1 \strokec4 \
\cb3 num_epochs = \cf6 \strokec6 100\cf0 \cb1 \strokec4 \
\cb3 batch_size = \cf6 \strokec6 128\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 for\cf0 \strokec4  epoch \cf7 \strokec7 in\cf0 \strokec4  \cf8 \strokec8 range\cf0 \strokec4 (num_epochs):\cb1 \
\cb3     X, y = generate_data_balanced(batch_size)\cb1 \
\cb3     input_sequences = X.tolist()\cb1 \
\
\cb3     \cf5 \strokec5 # Forward pass\cf0 \cb1 \strokec4 \
\cb3     logits = model(input_sequences).squeeze(\cf6 \strokec6 -1\cf0 \strokec4 )  \cf5 \strokec5 # shape: [batch_size]\cf0 \cb1 \strokec4 \
\cb3     loss = criterion(logits, y.\cf9 \strokec9 float\cf0 \strokec4 ())\cb1 \
\
\cb3     \cf5 \strokec5 # Backward pass\cf0 \cb1 \strokec4 \
\cb3     optimizer.zero_grad()\cb1 \
\cb3     loss.backward()\cb1 \
\cb3     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=\cf6 \strokec6 1.0\cf0 \strokec4 )\cb1 \
\cb3     optimizer.step()\cb1 \
\
\cb3     \cf5 \strokec5 # Accuracy\cf0 \cb1 \strokec4 \
\cb3     acc = ((logits > \cf6 \strokec6 0\cf0 \strokec4 ) == y).\cf9 \strokec9 float\cf0 \strokec4 ().mean().item()\cb1 \
\
\cb3     \cf2 \strokec2 if\cf0 \strokec4  epoch % \cf6 \strokec6 5\cf0 \strokec4  == \cf6 \strokec6 0\cf0 \strokec4 :\cb1 \
\cb3         \cf8 \strokec8 print\cf0 \strokec4 (\cf7 \strokec7 f\cf10 \strokec10 "[Epoch \cf0 \strokec4 \{epoch\}\cf10 \strokec10 ] Loss: \cf0 \strokec4 \{loss.item()\cf6 \strokec6 :.4f\cf0 \strokec4 \}\cf10 \strokec10  Acc: \cf0 \strokec4 \{acc\cf6 \strokec6 :.4f\cf0 \strokec4 \}\cf10 \strokec10 "\cf0 \strokec4 )\cb1 \
}