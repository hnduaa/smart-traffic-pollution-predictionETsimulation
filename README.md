# smart-traffic-pollution-predictionETsimulation
 Un système intelligent pour la prédiction des congestions routières et des niveaux de pollution, avec optimisation et simulation des feux tricolores. Ce projet utilise des algorithmes de machine learning et d’optimisation pour analyser le trafic, prévoir les embouteillages et la pollution, et ajuster dynamiquement les cycles des feux tricolores. Une simulation de trafic avec SUMOest intégrée pour valider les recommandations. , avec une visualisation interactive des résultats.


 to run the app : streamlit run main.py
to run the simulation :
1. cd sumo
2. run the 2 cmnds : 
netconvert --node-files=nodes.xml --edge-files=edges.xml --output=network.net.xml --tls.green.time 20 --tls.yellow.time 10 --tls.red.time 90
sumo-gui -c sumo.cfg