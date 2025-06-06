{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red131\green0\blue165;\red245\green245\blue245;\red0\green0\blue0;
\red0\green0\blue255;\red86\green65\blue25;\red0\green0\blue109;\red144\green1\blue18;\red31\green99\blue128;
\red19\green85\blue52;\red15\green112\blue1;}
{\*\expandedcolortbl;;\cssrgb\c59216\c13725\c70588;\cssrgb\c96863\c96863\c96863;\cssrgb\c0\c0\c0;
\cssrgb\c0\c0\c100000;\cssrgb\c41569\c32157\c12941;\cssrgb\c0\c6275\c50196;\cssrgb\c63922\c8235\c8235;\cssrgb\c14510\c46275\c57647;
\cssrgb\c6667\c40000\c26667;\cssrgb\c0\c50196\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import\cf0 \strokec4  torch\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 def\cf0 \strokec4  \cf6 \strokec6 is_good_number\cf0 \strokec4 (\cf7 \strokec7 digits\cf0 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3     \cf8 \strokec8 """Return 1 if replacing any digit with 1 results in a number divisible by 7."""\cf0 \cb1 \strokec4 \
\cb3     s = \cf8 \strokec8 ''\cf0 \strokec4 .join(\cf9 \strokec9 str\cf0 \strokec4 (d) \cf2 \strokec2 for\cf0 \strokec4  d \cf5 \strokec5 in\cf0 \strokec4  digits)\cb1 \
\cb3     \cf2 \strokec2 for\cf0 \strokec4  i \cf5 \strokec5 in\cf0 \strokec4  \cf6 \strokec6 range\cf0 \strokec4 (\cf10 \strokec10 4\cf0 \strokec4 ):\cb1 \
\cb3         modified = \cf9 \strokec9 list\cf0 \strokec4 (s)\cb1 \
\cb3         modified[i] = \cf8 \strokec8 '1'\cf0 \cb1 \strokec4 \
\cb3         num = \cf9 \strokec9 int\cf0 \strokec4 (\cf8 \strokec8 ''\cf0 \strokec4 .join(modified))\cb1 \
\cb3         \cf2 \strokec2 if\cf0 \strokec4  num % \cf10 \strokec10 7\cf0 \strokec4  != \cf10 \strokec10 0\cf0 \strokec4 :\cb1 \
\cb3             \cf2 \strokec2 return\cf0 \strokec4  \cf10 \strokec10 0\cf0 \cb1 \strokec4 \
\cb3     \cf2 \strokec2 return\cf0 \strokec4  \cf10 \strokec10 1\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 def\cf0 \strokec4  \cf6 \strokec6 generate_data_balanced\cf0 \strokec4 (\cf7 \strokec7 num_samples\cf0 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3     inputs = []\cb1 \
\cb3     targets = []\cb1 \
\
\cb3     \cf11 \strokec11 # Generate both positive and negative samples\cf0 \cb1 \strokec4 \
\cb3     positives = \cf10 \strokec10 0\cf0 \cb1 \strokec4 \
\cb3     negatives = \cf10 \strokec10 0\cf0 \cb1 \strokec4 \
\cb3     \cf2 \strokec2 while\cf0 \strokec4  \cf6 \strokec6 len\cf0 \strokec4 (inputs) < num_samples:\cb1 \
\cb3         digits = [torch.randint(\cf10 \strokec10 0\cf0 \strokec4 , \cf10 \strokec10 10\cf0 \strokec4 , ()).item() \cf2 \strokec2 for\cf0 \strokec4  _ \cf5 \strokec5 in\cf0 \strokec4  \cf6 \strokec6 range\cf0 \strokec4 (\cf10 \strokec10 4\cf0 \strokec4 )]\cb1 \
\cb3         \cf2 \strokec2 if\cf0 \strokec4  digits[\cf10 \strokec10 0\cf0 \strokec4 ] == \cf10 \strokec10 0\cf0 \strokec4 :  \cf11 \strokec11 # skip non-4-digit numbers\cf0 \cb1 \strokec4 \
\cb3             \cf2 \strokec2 continue\cf0 \cb1 \strokec4 \
\
\cb3         label = is_good_number(digits)\cb1 \
\cb3         \cf2 \strokec2 if\cf0 \strokec4  label == \cf10 \strokec10 1\cf0 \strokec4  \cf5 \strokec5 and\cf0 \strokec4  positives < num_samples // \cf10 \strokec10 2\cf0 \strokec4 :\cb1 \
\cb3             inputs.append(digits)\cb1 \
\cb3             targets.append(\cf10 \strokec10 1\cf0 \strokec4 )\cb1 \
\cb3             positives += \cf10 \strokec10 1\cf0 \cb1 \strokec4 \
\cb3         \cf2 \strokec2 elif\cf0 \strokec4  label == \cf10 \strokec10 0\cf0 \strokec4  \cf5 \strokec5 and\cf0 \strokec4  negatives < num_samples // \cf10 \strokec10 2\cf0 \strokec4 :\cb1 \
\cb3             inputs.append(digits)\cb1 \
\cb3             targets.append(\cf10 \strokec10 0\cf0 \strokec4 )\cb1 \
\cb3             negatives += \cf10 \strokec10 1\cf0 \cb1 \strokec4 \
\
\cb3     \cf2 \strokec2 return\cf0 \strokec4  torch.tensor(inputs), torch.tensor(targets)\cb1 \
}