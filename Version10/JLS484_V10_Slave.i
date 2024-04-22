SO 3887 JLS484 Replacement, Inventor Version 10
c -------------------------------------------------------------------
c
c   Replacement for the JLS484 irradiator at John Hopkins
c   Similar Geometry, 26000 Ci of Co60 2 source rods 4 pellets
c   
c   Reminder:
c   Currently modeling the main shield and specifically the beam chamber
c   have not added the angled side to the beam chanber
c
c -------------------------------------------------------------------
c
c                       Cell Cards
c
c -------------------------------------------------------------------
c       Source Rods
c -------------------------------------------------------------------
c
1 1 -4.43 -100:-101:-108:-109 $ Sources
2 100 -4.0 (-102 100):(-103 101):(-110 108):(-111 109):-104:-105:-112:-113 $ Cladding
3 100 -4.0 (-107 106):(-115 114):-128:-129:-130:-132:-133:-134: $ SS Source Rod
        (-124 119):(-125 116 117 118 119):(-126 123):(-127 120 121 122 123)
4 300 -4.2 (-131 130):(-135 134) $ Bronze Bearing
5 200 -9.65 -136:-137:-138:-139:-140:-141:-142:-143:-144:-145 $ Lower Tungsten Rod
6 300 -4.2 (-146 140):(-147 145) $Bronze Bearing
7 200 -9.65 -148:-149:-150:-151:-152:-153
c -------------------------------------------------------------------
c       Primary Shield
c -------------------------------------------------------------------
10 100 -4.0 (-201 200):(-203 202) $ Source Rod Tubes
11 99 -0.0006025 -204 -206 -207 201 203 $Inside Beam Chamber
12 100 -4.0 (-205 -208 -209) (204:206:207) 201 203 $SS Chamber
13 400 -5.675 (-210 -211 -212 -213 -214 -215 -216 201 203 ):(-225 -213 -214 -215 -216 201 203) #11 #12 $ Pb Main
14 100 -4.0 (-217 -218 -219 -220 -221 -222 -223 201 203 #13):(-226 -218 -219 -220 -221 -222 -223 201 203 #11 #12 #13)$ SS Main
15 100 -4.0 (-224 -218 -219)
c -------------------------------------------------------------------
c       Secondary Shield
c -------------------------------------------------------------------
20 400 -5.675 (-250 -251 -252 -253 -254 #23 #22 270 #25 #26 #27):(-255 #23 #22 272 #25 #26 #27 291 #28 #29) $ Lead
21 100 -4.0 (-256 -257 -258 -259 -260 #20 #22 270 #25 #26 #27):(-261 #20 #22 272 #25 #25 #27 291 #28 #29) $ Steel Liner
22 99 -0.0006025 -265 #25 #26 #27 $ Air in Chamber
23 100 -4.0 -266 265 270 272 #25 #26 #27 #28 #29 $ Stell Chamber Liner
24 100 -4.0 (-270 271 -260):(-272 273) $ Cable Ports
25 99 -0.0006025 (-276 278):(-280 282):(-284)$ Door Space
26 100 -4.0 (-275 277 #25):(-279 281 #25):(-283 -259 #25) $ side Door Liner
27 100 -4.0 (-290 291 265) #28$ Front Door
28 99 -0.0006025 (-293 295):(-292 -294)
29 100 -4.0 (-292 294 290) #28
999 0 999

c -------------------------------------------------------------------
c
c                       Surface Cards
c
c -------------------------------------------------------------------
c       Source Rods
c -------------------------------------------------------------------
100 RCC 0 4.45 2.959 0 0 2.972 0.9905   $ +Y/+Z Source
101 RCC 0 4.45 -2.959 0 0 -2.972 0.9905 $ +Y/-Z Source
102 RCC 0 4.45 2.807 0 0 3.680 1.1735
103 RCC 0 4.45 -2.403 0 0 -3.680 1.1735
104 RCC 0 4.45 0 0 0 2.807 1.1735
105 RCC 0 4.45 0 0 0 -2.403 1.1735
106 RCC 0 4.45 -6.159 0 0 13.122 1.186
107 RCC 0 4.45 -6.952 0 0 13.914 1.3385
c
108 RCC 0 -4.45 2.959 0 0 2.972 0.9905   $ -Y/+Z Source
109 RCC 0 -4.45 -2.959 0 0 -2.972 0.9905 $ -Y/-Z Source
110 RCC 0 -4.45 2.807 0 0 3.680 1.1735
111 RCC 0 -4.45 -2.403 0 0 -3.680 1.1735
112 RCC 0 -4.45 0 0 0 2.807 1.1735
113 RCC 0 -4.45 0 0 0 -2.403 1.1735
114 RCC 0 -4.45 -6.159 0 0 13.122 1.186
115 RCC 0 -4.45 -6.952 0 0 13.914 1.3385
c
c Modelling point where tungsten meets the source rod
c Interior
116 TRC 0 4.45 7.312 0 0 -0.076 1.2625 1.186 
117 RCC 0 4.45 7.312 0 0 0.318 1.4455
118 RCC 0 4.45 7.63 0 0 0.889 1.3145
119 RCC 0 4.45 6.487 0 0 0.749 1.186
c
120 TRC 0 -4.45 7.312 0 0 -0.076 1.2625 1.186 
121 RCC 0 -4.45 7.312 0 0 0.318 1.4455
122 RCC 0 -4.45 7.63 0 0 0.889 1.3145
123 RCC 0 -4.45 6.487 0 0 0.749 1.186
c Exterior
124 TRC 0 4.45 7.122 0 0 -0.16 1.778 1.3385
125 RCC 0 4.45 7.122 0 0 1.397 1.778
c
126 TRC 0 -4.45 7.122 0 0 -0.16 1.778 1.3385 
127 RCC 0 -4.45 7.122 0 0 1.397 1.778
c
c Bottom of Source Rod
128 TRC 0 4.45 -7.112 0 0 0.16 1.778 1.3385
129 RCC 0 4.45 -7.112 0 0 -0.318 1.778
130 RCC 0 4.45 -7.43 0 0 -1.27 0.635
131 RCC 0 4.45 -7.34 0 0 -1.27 1.778
c
132 TRC 0 -4.45 -7.112 0 0 0.16 1.778 1.3385
133 RCC 0 -4.45 -7.112 0 0 -0.318 1.778
134 RCC 0 -4.45 -7.43 0 0 -1.27 0.635
135 RCC 0 -4.45 -7.34 0 0 -1.27 1.778
c
c Tungsten Rods
c Lower 
136 RCC 0 4.45 6.487 0 0 0.876 1.1225
137 RCC 0 4.45 7.363 0 0 0.267 1.4265
138 RCC 0 4.45 7.63 0 0 0.889 1.3145
139 RCC 0 4.45 8.519 0 0 11.991 1.829
140 RCC 0 4.45 20.51 0 0 3.81 0.9525
c
141 RCC 0 -4.45 6.487 0 0 0.876 1.1225
142 RCC 0 -4.45 7.363 0 0 0.267 1.4265
143 RCC 0 -4.45 7.63 0 0 0.889 1.3145
144 RCC 0 -4.45 8.519 0 0 11.991 1.829
145 RCC 0 -4.45 20.51 0 0 3.81 0.9525
c Bearing
146 RCC 0 4.45 20.51 0 0 1.27 1.9175
147 RCC 0 -4.45 20.51 0 0 1.27 1.9175
c Upper 
148 RCC 0 4.45 21.78 0 0 3.81 1.829
149 RCC 0 4.45 25.59 0 0 20.955 1.829
150 RCC 0 4.45 46.545 0 0 5.715 1.1115
c
151 RCC 0 -4.45 21.78 0 0 3.81 1.829
152 RCC 0 -4.45 25.59 0 0 20.955 1.829
153 RCC 0 -4.45 46.545 0 0 5.715 1.1115
c -------------------------------------------------------------------
c       Air Cylinder Tower
c -------------------------------------------------------------------
c -------------------------------------------------------------------
c       Primary Shield
c -------------------------------------------------------------------
200 RCC 0 4.45 -55.055 0 0 104.776 1.918
201 RCC 0 4.45 -55.36 0 0 105.081 2.223
202 RCC 0 -4.45 -55.055 0 0 104.776 1.918
203 RCC 0 -4.45 -55.36 0 0 105.081 2.223
c
c Beam Chamber
204 RPP -2.223 29.21 -13.335 13.335 -14.605 14.605
205 RPP -3.35 29.21 -13.97 13.97 -15.24 15.24
c For the Planes Inner_Y= 1.4475+W, Outer_Y=1.744+W
c P X1 Y1 Z1  X2 Y2 Z2  X3 Y3 Z3
206 P 7.99 13.335 14.605 -2.223 5.8975 14.605 -2.223 5.8975 -14.605
207 P 7.99 -13.335 14.605 -2.223 -5.8975 14.605 -2.223 -5.8975 -14.605
208 P 7.99 13.97 15.24 -2.858 6.194 15.24 -2.858 6.194 -15.24
209 P 7.99 -13.97 15.24 -2.858 -6.194 15.24 -2.858 -6.194 -15.24
c
c Lower Section
c Lead
210 RPP -32.703 32.068 -41.91 41.91 -83.934 -44.133
211 P 32.068 6.105 -44.133 32.068 6.105 -83.934 28.352 41.91 -83.934
212 P 32.068 -6.105 -44.133 32.068 -6.105 -83.934 28.352 -41.91 -83.934
213 P -5.011 41.91 -44.133 -5.011 41.91 -83.934 -22.543 28.05 -83.934
214 P -5.011 -41.91 -44.133 -5.011 -41.91 -83.934 -22.543 -28.05 -83.934
215 P -22.543 28.05 -44.133 -32.703 14.191 -44.133 -32.703 14.191 -83.934
216 P -22.543 -28.05 -44.133 -32.703 -14.191 -44.133 -32.703 -14.191 -83.934
c
c Stainless Steel
217 RPP -33.655 33.020 -42.8625 42.8625 -86.475 -44.133
218 P 33.02 6.154 -44.133 33.02 6.154 -86.475 29.21 42.8625 -86.475
219 P 33.02 -6.154 -44.133 33.02 -6.154 -86.475 29.21 -42.8625 -86.475
220 P -5.267 42.8625 -44.133 -23.24 28.654 -44.133 -23.24 28.654 -86.475
221 P -5.267 -42.8625 -44.133 -23.24 -28.654 -44.133 -23.24 -28.654 -86.475
222 P -23.24 28.654 -44.133 -23.24 28.654 -86.475 -33.655 14.446 -86.475
223 P -23.24 -28.654 -44.133 -23.24 -28.654 -86.475 -33.655 -14.446 -86.475
224 RPP 28.575 33.02 -42.8625 42.8625 -44.133 -43.498
c
c Upper Section
c Lead
225 RPP -32.703 28.575 -41.91 41.91 -44.133 45.276
c Steel
226 RPP -33.655 29.21 -42.8625 42.8625 -44.133 49.721
c -------------------------------------------------------------------
c       Secondary Shield        Sem is set at 1mm gap
c -------------------------------------------------------------------
c Lead Cone
250 RPP 29.945 79.464 -41.962 41.962 -42.863 48.724
251 P 29.576 41.962 48.724 29.576 -41.962 48.724 79.464 24.13 25.4 $ Top
252 P 33.006 41.962 -42.863 33.006 -41.962 -42.863 79.464 24.13 -25.4 $ Bottom
253 P 29.576 41.962 48.724 29.576 41.962 -42.863 79.464 24.13 -25.4 $ Side
254 P 29.576 -41.962 48.724 29.576 -41.962 -42.863 79.464 -24.13 -25.4 $ Side
c
255 RPP 79.464 112.315 -24.13 24.13 -25.4 25.4
c Steel Shell
256 RPP 29.31 80.11 -47.9425 47.9425 -43.498 54.801
257 P 30.58 -42.41 49.128 30.58 42.41 49.128 80.11 24.765 26.035 $ Top
258 P 33.12 -42.41 -43.498 33.12 42.41 -43.498 80.11 24.765 -26.035 $ Top
259 P 30.58 -42.41 49.128 30.58 -42.14 -43.498 80.11 -24.765 26.035 $ Side
260 P 30.58 42.41 49.128 30.58 42.14 -43.498 80.11 24.765 26.035 $ Side
c
261 RPP 80.11 113.13 -24.765 24.765 -26.035 26.035
c
c Beam Chamber
265 RPP 29.31 107.415 -13.335 13.335 -14.605 14.605
266 RPP 29.31 107.415 -13.97 13.97 -15.24 15.24
c 
c Cable Ports
270 RCC 38.2 13.335 0 0 48 0 4.445  $ Lead
271 RCC 38.2 13.335 0 0 40 0 3.81  $ Lead
272 RCC 95.57 0 14.605 0 0 12.79 4.445
273 RCC 95.57 0 14.605 0 0 12.79 3.810
c
c Side Door Port
275 RPP 40.295 71.855 -17.78 -13.335 -10.795 10.795
276 RPP 40.74 71.22 -18.415 -13.335 -10.16 10.16
277 P 41.502 -13.97 -10.795 41.502 -13.97 10.795 40.295 -17.78 10.795
278 P 43.039 -13.335 -10.16 43.039 -13.335 10.16 40.74 -18.415 -10.16
c
279 RPP 36.485 75.665 -21.59 -17.78 -14.605 14.605
280 RPP 36.93 75.03 -23.007 -18.415 -13.97 13.97
281 P 37.872 -17.78 -14.605 37.872 -17.78 14.605 36.485 -21.59 14.605
282 P 38.317 -18.415 -13.97 38.317 -18.415 13.97 36.485 -23.007 13.97
c
283 RPP 32.305 79.295 -45.505 -21.59 -18.415 18.415
284 RPP 33.12 78.66 -45.505 -23.007 -17.78 17.78
c
c Front Door Port
290 RPP 106.78 113.13 -21.59 21.59 -22.86 22.86
291 RPP  107.415 113.13 -20.955 20.955 -22.225 22.225
c Cable Pass
292 RPP 97.55 113.13 -4.445 4.445 -25.7 -15.54
293 RPP 97.846 113.13 -3.81 3.81 -26.035 -14.605
294 P 97.55 -4.445 -15.54 97.55 4.445 -15.54 106.075 -4.445 -25.7 
295 P 97.846 -3.81 -14.605 97.846 3.81 -14.605 107.415 3.81 -26.035
296 RPP 106.78 113.13 -3.81 3.81 -26.035 -14.605
297 P 107.415 133.13 -3.81 3.81 
c
999 RPP -100 100 -100 100 -100 100      $ Kill Volume

c -------------------------------------------------------------------
c
c                       Data Cards
c
c -------------------------------------------------------------------
MODE P
NPS 3.7E10
PRINT 10 60 110
PRDMP J 5e7 0 1
RAND 97008386995783
IMP:P 1 23R 0
c -------------------------------------------------------------------
c           Material Cards
c -------------------------------------------------------------------
m1 27000 -1         $ Nat-Co
c
c   
m100 $$ Stainless Steel 316, p=8.0 PNNL-15870
        6000 -0.0008
        25055 -0.02
        15031 -0.00045
        16000 -0.0003
        14000 -0.01
        24000 -0.17
        28000 -0.12
        42000 -0.025
        26000 -0.65345
m200 $$ Tungsten, p=19.3 
        74000 -1.0
m300 $$ Bronze, p=8.4 PNNL-15870
        13027 -0.028528
        14000 -0.003339
        25055 -0.003555
        26000 -0.010208
        28000 -0.006718
        29000 -0.874155
        30000 -0.036037
        50000 -0.024503
        82000 -0.012957
c
m400 $$ Lead, p=11.35
        82000  -1.0
c
m99 $ Dry Air, p=0.001205 PNNL-15870
        6000 -0.000124
        7014 -0.752316
        7015 -0.002944
        8016 -0.231153
        8017 -0.000094
        8018 -0.000535
        18000 -0.012827
c -------------------------------------------------------------------
c       SDEF Card
c -------------------------------------------------------------------
SDEF PAR=2
     ERG=D1
     POS=D2
     AXS=0 0 1
     RAD=D3
     EXT=D4
c
c
SI1 L 1.1732  1.3325	$ Cobalt-60 gammas
SP1 D 0.49969 0.50031
c
SI2 L 0 4.45 4.445  0 -4.45 4.445  0 4.45 -4.445  0 -4.45 -4.445
SP2 0.25 0.25 0.25 0.25		$ Seting Center of Source Points
c
SI3 L 0 0.9906	 $ Isotropic Radial Sampling
SP3 -21 1
c
SI4 L -1.4859 1.4869	$ Isotopic Axial Sampling	
SP4 -21 0