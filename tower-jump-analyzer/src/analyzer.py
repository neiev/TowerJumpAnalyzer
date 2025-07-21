import csv
from datetime import datetime
import os
import re
from turtle import shape
from shapely import Point
from shapely.geometry import shape
import pandas as pd

class TowerJumpAnalyzer:
    def __init__(self):
        # Configurações principais
        self.min_time_diff_threshold = 5 * 60  # 5 minutos em segundos
        self.max_time_diff_threshold = 60 * 60  # 1 hora em segundos
        self.max_velocity_threshold = 900  # max km/h para considerar um tower jump
        self.max_jump_distance = 300  # Distância máxima para considerar um tower jump (em km)
        self.min_jump_distance = 10  # Distância mínima para considerar um tower jump (em km)

        self.date_format = '%m/%d/%y %H:%M'
        self.min_confidence = 0
        self.state_pattern = re.compile(r',\s*([A-Z]{2}|[A-Za-z\s]+)$')
        self.us_state_abbreviations = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 
            'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 
            'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
            'MD': 'Maryland', 'MA': 'Massachusetts', 
            'MI': 'Michigan', 'MN': 'Minnesota', 
            'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota',
            'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
            'PA': 'Pennsylvania', 'RI': 'Rhode Island',
            'SC': 'South Carolina', 'SD': 'South Dakota',
            'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 
            'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        
        # Carrega os polígonos dos estados (geojson simplificado)
        self.load_state_geometries()
        
        # Prioridade de tipos de célula
        self.cell_type_priority = {
            'macro': 1,
            'small': 2,
            'micro': 3,
            'pico': 4,
            'femto': 5,
            'unknown': 0
        }

        # Limiares para resolução de conflitos
        self.min_duration_to_override = 3  # minutos
        self.min_confidence_diff = 20  # porcentagem
        self.min_confidence_absolute = 70  # porcentagem mínima

    def safe_float(self, value):
        """Converte string para float, retornando None se falhar"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def parse_datetime(self, datetime_str):
        """Converte string de data/hora para objeto datetime"""
        try:
            return datetime.strptime(datetime_str, self.date_format)
        except ValueError as e:
            print(f"Erro ao analisar data/hora: {str(e)}")
            return None

    def calculate_location_score(self, entry):
        """Calcula o score de localização baseado em confiança e tipo de célula"""
        score = 0
        if entry['confidence'] >= self.min_confidence:
            score += entry['confidence']
        
        cell_type = entry['cell_types'].lower().strip()
        if cell_type in self.cell_type_priority:
            score += (100 - self.cell_type_priority[cell_type] * 10)

        if entry['state'] == 'UNKNOWN':
            score -= 20

        if entry['latitude'] is None or entry['longitude'] is None:
            score -= 30

        if entry.get('tower_jump', False):
            score -= 50
        
        score = max(score, 0)
        return min(score, 100)

    def load_state_geometries(self):
        """Carrega os polígonos dos estados a partir de um GeoJSON simplificado"""
        try:
            states_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "New York"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-79.8, 40.5], [-71.8, 40.5], [-71.8, 45.1], [-79.8, 45.1], [-79.8, 40.5]]]
                        }
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Connecticut"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-73.7, 40.9], [-71.8, 40.9], [-71.8, 42.1], [-73.7, 42.1], [-73.7, 40.9]]]
                        }
                    },
                    # Adicione outros estados conforme necessário
                ]
            }

            self.state_geometries = {}
            for feature in states_geojson['features']:
                state_name = feature['properties']['name']
                self.state_geometries[state_name] = shape(feature['geometry'])  # ✅ CONVERSÃO CORRETA

        except Exception as e:
            print(f"Erro ao carregar geometrias dos estados: {str(e)}")
            self.fallback_state_detection = True
        else:
            self.fallback_state_detection = False

    def extract_state(self, location):
        """Extrai o estado da string de localização com mais robustez"""
        if not location:
            return 'UNKNOWN'
        
        location = re.sub(r'[^\w\s,]', '', location).strip().upper()
        
        # 1. Verifica sigla de estado
        state_abbr = re.search(r'\b([A-Z]{2})\b(?!\s*\d)', location)
        if state_abbr and state_abbr.group(1) in self.us_state_abbreviations:
            return self.us_state_abbreviations[state_abbr.group(1)]
        
        # 2. Verifica nomes completos
        for state_abbr, state_name in self.us_state_abbreviations.items():
            if re.search(r'\b' + re.escape(state_name.upper()) + r'\b', location):
                return state_name
            
            if state_name == 'New York' and 'NY' in location:
                return state_name
            if state_name == 'California' and 'CA' in location:
                return state_name
        
        # 3. Verifica padrões comuns
        match = re.search(r',\s*([A-Z]{2}|[A-Za-z\s]+)$', location)
        if match:
            state_candidate = match.group(1).title()
            if state_candidate in self.us_state_abbreviations.values():
                return state_candidate
            if len(state_candidate) == 2 and state_candidate.upper() in self.us_state_abbreviations:
                return self.us_state_abbreviations[state_candidate.upper()]
        
        # 4. Verifica referências regionais
        ny_keywords = ['NYC', 'NEW YORK', 'MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND']
        if any(keyword in location for keyword in ny_keywords):
            return 'New York'
        
        ca_keywords = ['LOS ANGELES', 'SAN FRANCISCO', 'SAN DIEGO', 'CALIFORNIA']
        if any(keyword in location for keyword in ca_keywords):
            return 'California'
        
        return 'UNKNOWN'

    def precise_state_from_coordinates(self, lat, lon):
        """Determina o estado usando coordenadas com fallback robusto"""
        if lat is None or lon is None or lat == 0.0 or lon == 0.0:
            return 'UNKNOWN'
        
        point = Point(lon, lat)
        state = 'UNKNOWN'
        
        if not self.fallback_state_detection:
            for state_name, geometry in self.state_geometries.items():
                if geometry.contains(point):
                    state = state_name
                    break
        
        if state == 'UNKNOWN':
            state = self.infer_state_from_coordinates(lat, lon)
        
        if state == 'UNKNOWN':
            state = self.check_coastal_areas(lat, lon)
        
        return state

    def check_coastal_areas(self, lat, lon):
        """Verifica áreas costeiras onde os estados podem se sobrepor"""
        # Florida
        if (24.5 <= lat <= 31.0) and (-87.6 <= lon <= -80.0):
            return 'Florida'
        
        # New York area
        if (40.5 <= lat <= 45.0) and (-79.8 <= lon <= -71.8):
            return 'New York'
        
        # California
        if (32.5 <= lat <= 42.0) and (-124.5 <= lon <= -114.0):
            return 'California'
        
        # Texas
        if (25.8 <= lat <= 36.5) and (-106.7 <= lon <= -93.5):
            return 'Texas'
        
        return 'UNKNOWN'

    def get_current_location(self, data):
        """Obtém a localização atual com base nos dados mais recentes"""
        if not data:
            return {'status': 'NO_DATA', 'state': 'UNKNOWN', 'coordinates': None, 'confidence': 0, 'location_quality': 0}
        
        latest_entry = max(data, key=lambda x: x['start_time'])
        
        if latest_entry['state'] == 'UNKNOWN':
            return {
                'status': 'NO_VALID_DATA',
                'state': latest_entry['state'],
                'coordinates': (latest_entry['latitude'], latest_entry['longitude']),
                'confidence': latest_entry['confidence'],
                'location_quality': latest_entry['location_score'],
                'location_score': latest_entry['location_score'],
                'last_update': latest_entry['start_time'],
                'minutes_since_update': 0,
                'tower_jump': latest_entry.get('tower_jump', False),
                'cell_type': latest_entry.get('cell_types', '')
            }
        
        return {
            'status': 'CURRENT',
            'state': latest_entry['state'],
            'coordinates': (latest_entry['latitude'], latest_entry['longitude']),
            'confidence': latest_entry['confidence'],
            'location_quality': latest_entry['location_score'],
            'location_score': latest_entry['location_score'],
            'last_update': latest_entry['start_time'],
            'minutes_since_update': (latest_entry['start_time'] - data[0]['start_time']).total_seconds() / 60,
            'tower_jump': latest_entry.get('tower_jump', False),
            'cell_type': latest_entry.get('cell_types', '')
        }

    def resolve_state_conflicts(self, data):
        """Resolve conflitos de estado entre registros"""
        if not data:
            return data
        
        resolved_data = []
        for i in range(len(data)):
            current = data[i]
            if i == 0 or current['state'] != data[i - 1]['state']:
                resolved_data.append(current)
                continue
            
            previous = data[i - 1]
            time_diff = (current['start_time'] - previous['end_time']).total_seconds() / 60
            
            if time_diff < self.min_duration_to_override:
                if (current['confidence'] >= self.min_confidence_absolute and 
                    current['confidence'] > previous['confidence'] + self.min_confidence_diff):
                    current['conflict_resolution'] = 'OVERRIDE'
                    current['discarded_state'] = previous['state']
                    current['resolved_by'] = 'STATE_CONFLICT_RESOLUTION'
                else:
                    current['conflict_resolution'] = 'NO_CONFLICT'
                    current['discarded_state'] = None
                    current['resolved_by'] = None
            else:
                current['conflict_resolution'] = 'NO_CONFLICT'
                current['discarded_state'] = None
                current['resolved_by'] = None
            
            resolved_data.append(current)
        
        return resolved_data

    def calculate_velocity(self, point1, point2, time_diff):
        """Calcula a velocidade entre dois pontos em km/h"""
        if point1 is None or point2 is None or time_diff <= 0:
            return 0
        
        distance = point1.distance(point2) * 111  # Distância em km
        velocity = distance / (time_diff / 3600)  # km/h
        return velocity

    def detect_vehicle_type(self, velocity, distance, time_diff):
        """Detecta o tipo de veículo com base na velocidade, tempo e distância"""
        if velocity <= 0:
            return 'UNKNOWN'
        
        vehicle_types = {
            'Avião': (200, 900),
            'Carro': (20, 200),
            'Barco': (5, 100),
            'Trem': (30, 300),
            'Ônibus': (10, 120)
        }
        
        for vehicle, (min_speed, max_speed) in vehicle_types.items():
            if min_speed <= velocity <= max_speed:
                if (velocity * (time_diff / 3600)) >= distance:
                    return vehicle
                    
        return 'UNKNOWN'

    def criar_objeto_padrao(self, data):
        objeto = {
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'start_time': data['start_time'],
            'end_time': data['end_time'],
            'vehicle_type': data.get('vehicle_type', 'UNKNOWN'),
            'location_score': data.get('location_score', 0),
            'cell_types': data.get('cell_types', ''),
            'confidence': data.get('confidence', 0),
            'tower_jump': data.get('tower_jump', False),
            'conflict_resolution': data.get('conflict_resolution', 'NO_CONFLICT'),
            'discarded_state': data.get('discarded_state', None),
            'resolved_by': data.get('resolved_by', None),
            'distance': data.get('distance', 0),
            'velocity': data.get('velocity', 0),
            'is_location_change_possible': data.get('is_location_change_possible', False),
            'duration': data.get('duration', 0),
            'same_time_diff_state': data.get('same_time_diff_state', 'DIFFERENT_TIME_DIFF_STATE'),
            'state': data.get('state', 'UNKNOWN'),
            'page': data.get('page', ''),
            'item': data.get('item', '')
        }
        return objeto

    def marcar_tower_jump(self, current, previous):
        current['tower_jump'] = True
        current['conflict_resolution'] = 'TOWER_JUMP'
        current['discarded_state'] = previous['state'] 
        current['resolved_by'] = 'TOWER_JUMP_DETECTION'
        previous['conflict_resolution'] = 'TOWER_JUMP'
        previous['discarded_state'] = previous['state'] 
        previous['resolved_by'] = 'TOWER_JUMP_DETECTION'
        return current, previous

    def detect_jumps(self, data):
        """Detecta tower jumps com base nos dados de localização"""
        if not data:
            return data
        
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i - 1]
            
            if not current['start_time'] or not previous['start_time']:
                continue

            if current['latitude'] is None or current['longitude'] is None:
                continue

            time_diff = (current['start_time'] - previous['end_time']).total_seconds()

            previous_state = previous['state']

            if previous_state == 'UNKNOWN':
                valid_previous_state = 'UNKNOWN'
                while i > 1 and valid_previous_state == 'UNKNOWN':
                    i -= 1
                    if data[i]['state'] and data[i]['state'] != 'UNKNOWN':
                        valid_previous_state = data[i]['state']      
                if valid_previous_state == 'UNKNOWN':
                    continue

                previous['state'] = valid_previous_state
                previous['latitude'] = data[i]['latitude']
                previous['longitude'] = data[i]['longitude']
                previous['start_time'] = data[i]['start_time']
                previous['end_time'] = data[i]['end_time']
                previous['page'] = data[i].get('page', '')
                previous['item'] = data[i].get('item', '')
            
            current_state = current['state']

            if previous_state == current_state:
                if previous['start_time'] == current['start_time']:
                    current['same_time_diff_state'] = 'SAME_TIME_DIFF_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFFERENT_TIME_DIFF_STATE'

            if previous['latitude'] is None and previous['longitude'] is None:
                previous = self.criar_objeto_padrao(previous)
                continue
                
            if current['latitude'] is None or current['longitude'] is None:
                current = self.criar_objeto_padrao(current)
                continue

            if current_state == 'UNKNOWN':
                current = self.criar_objeto_padrao(current)
                
            if (previous['latitude'] is None and previous['longitude'] is None):
                if previous['state'] == 'UNKNOWN':
                    current = self.criar_objeto_padrao(current)
                    continue

            current_point = Point(current['longitude'], current['latitude'])
            previous_point = Point(previous['longitude'], previous['latitude'])
            
            velocity = self.calculate_velocity(
                previous_point,
                current_point,
                time_diff
            )
            
            current['vehicle_type'] = self.detect_vehicle_type(velocity, current.get('distance', 0), time_diff)
            current['velocity'] = velocity
            previous['velocity'] = previous.get('velocity', 0)

            current['location_score'] = self.calculate_location_score(current)
            previous['location_score'] = self.calculate_location_score(previous)

            current['same_time_diff_state'] = (current['state'] == previous['state'])
            current['duration'] = (current['start_time'] - previous['end_time']).total_seconds() / 60

            distance = current_point.distance(previous_point) * 111  # Distância em km
            current['distance'] = distance

            is_velocyty_high = velocity > self.max_velocity_threshold

            if is_velocyty_high:
                current['conflict_resolution'] = 'HIGH_VELOCITY'
                self.marcar_tower_jump(current, previous)
                continue

            if current['state'] != previous['state']:
                if (velocity > 0 and distance > self.min_jump_distance) or \
                    (is_velocyty_high and distance > self.min_jump_distance):
                    current['conflict_resolution'] = 'INCOMPATIBLE_DISTANCE'
                    self.marcar_tower_jump(current, previous)
                    continue

            if (current['state'] != previous['state'] and 
                current['same_time_diff_state'] and 
                time_diff < self.min_time_diff_threshold):
                current['conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
                self.marcar_tower_jump(current, previous)
                continue

            if (current['state'] != previous['state'] and 
                current['same_time_diff_state'] and 
                time_diff < self.max_time_diff_threshold):
                
                if (is_velocyty_high and distance > self.min_jump_distance) or \
                    (velocity > 0 and time_diff < self.min_time_diff_threshold and distance > self.min_jump_distance):
                    current['conflict_resolution'] = 'RAPID_STATE_JUMP'
                    self.marcar_tower_jump(current, previous)
                    continue

            if (current['state'] != previous['state'] and
                not current['same_time_diff_state']):
                if (is_velocyty_high and distance > self.min_jump_distance) or \
                    (velocity > 0 and time_diff < self.min_time_diff_threshold and distance > self.min_jump_distance):
                    current['conflict_resolution'] = 'INTERLEAVED_STATE_JUMP'
                    self.marcar_tower_jump(current, previous)
                    continue

            if (current['state'] == previous['state'] and 
                time_diff < self.min_time_diff_threshold):
                
                if (velocity > self.max_velocity_threshold and distance > self.min_jump_distance) or \
                    (velocity > 0 and time_diff < self.min_time_diff_threshold and distance > self.min_jump_distance):
                    current['conflict_resolution'] = 'CONSECUTIVE_STATE_JUMP'
                    self.marcar_tower_jump(current, previous)
                    continue

            if (current['state'] != previous['state'] and 
                current['same_time_diff_state'] and 
                time_diff < self.min_time_diff_threshold):
                current['conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
                self.marcar_tower_jump(current, previous)
                continue

            if (velocity > self.max_velocity_threshold and time_diff < self.min_time_diff_threshold) or \
                (velocity > 0 and time_diff > self.min_time_diff_threshold and
                (current['state'] != previous['state'] or current['same_time_diff_state'])):
                current['conflict_resolution'] = 'HIGH_VELOCITY_JUMP'
                self.marcar_tower_jump(current, previous)
                continue

            if time_diff > self.min_time_diff_threshold and distance > self.min_jump_distance:
                current['conflict_resolution'] = 'LONG_DISTANCE_JUMP'
                self.marcar_tower_jump(current, previous)
                continue
            
        return data

    def infer_state_from_coordinates(self, lat, lon):
        """Infere o estado com base nas coordenadas usando bounding boxes"""
        if lat is None or lon is None:
            return 'UNKNOWN'
        
        state_bounds = {
            'New York': {'lat': (40.5, 45.1), 'lon': (-79.8, -71.8)},
            'Connecticut': {'lat': (40.9, 42.1), 'lon': (-73.7, -71.8)},
            'New Jersey': {'lat': (38.9, 41.4), 'lon': (-75.6, -73.9)},
            'Pennsylvania': {'lat': (39.7, 42.3), 'lon': (-80.5, -74.7)},
            'Massachusetts': {'lat': (41.2, 42.9), 'lon': (-73.5, -69.9)},
            'Rhode Island': {'lat': (41.1, 42.0), 'lon': (-71.9, -71.1)},
        }
        
        candidate_states = []
        
        for state, bounds in state_bounds.items():
            if (bounds['lat'][0] <= lat <= bounds['lat'][1] and 
                bounds['lon'][0] <= lon <= bounds['lon'][1]):
                candidate_states.append(state)
        
        if len(candidate_states) == 1:
            return candidate_states[0]
        
        if len(candidate_states) > 1:
            small_states = {'Connecticut', 'Rhode Island', 'New Jersey'}
            for state in candidate_states:
                if state in small_states:
                    return state
            
            return self.resolve_border_conflict(lat, lon, candidate_states)
        
        return 'UNKNOWN'

    def resolve_border_conflict(self, lat, lon, candidate_states):
        """Resolve conflitos em áreas de fronteira entre estados"""
        state_reference_points = {
            'New York': (40.7, -74.0),
            'Connecticut': (41.6, -72.7),
            'New Jersey': (40.2, -74.7),
            'Pennsylvania': (40.3, -76.9),
            'Massachusetts': (42.4, -71.1),
            'Rhode Island': (41.8, -71.4),
        }
        
        distances = {}
        for state in candidate_states:
            if state in state_reference_points:
                ref_lat, ref_lon = state_reference_points[state]
                distance = ((lat - ref_lat)**2 + (lon - ref_lon)**2)**0.5
                distances[state] = distance
        
        if distances:
            return min(distances.items(), key=lambda x: x[1])[0]
        
        return candidate_states[0] if candidate_states else 'UNKNOWN'

    def fix_consecutive_unknowns(self, data):
        """Corrige sequências de estados UNKNOWN com base nos vizinhos conhecidos, exceto para registros com coordenadas inválidas"""
        if not data:
            return data
        
        for i in range(len(data)):
            # Não corrige registros com coordenadas inválidas
            if data[i]['latitude'] is None or data[i]['longitude'] is None:
                continue
                
            if data[i]['state'] == 'UNKNOWN':
                next_known = None
                for j in range(i+1, len(data)):
                    if data[j]['state'] != 'UNKNOWN' and data[j]['latitude'] is not None and data[j]['longitude'] is not None:
                        next_known = data[j]['state']
                        break
                
                prev_known = None
                for j in range(i-1, -1, -1):
                    if data[j]['state'] != 'UNKNOWN' and data[j]['latitude'] is not None and data[j]['longitude'] is not None:
                        prev_known = data[j]['state']
                        break
                
                if prev_known and next_known and prev_known == next_known:
                    data[i]['state'] = prev_known
                    data[i]['resolved_by'] = 'CONSECUTIVE_UNKNOWN_FIX'
                elif prev_known:
                    data[i]['state'] = prev_known
                    data[i]['resolved_by'] = 'PREVIOUS_KNOWN_FIX'
                elif next_known:
                    data[i]['state'] = next_known
                    data[i]['resolved_by'] = 'NEXT_KNOWN_FIX'
        
        return data

    def load_data(self, file_path):
        """Carrega e processa os dados do arquivo CSV com validação rigorosa"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                
                required_columns = {'UTCDateTime', 'LocalDateTime', 'State', 'Latitude', 'Longitude'}
                if not all(col in reader.fieldnames for col in required_columns):
                    missing = required_columns - set(reader.fieldnames)
                    print(f"Erro: Colunas obrigatórias faltando: {missing}")
                    return []
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Parse básico
                        datetime_str = row.get('UTCDateTime') or row.get('LocalDateTime')
                        start_time = self.parse_datetime(datetime_str) or datetime.min
                        end_time = start_time
                        
                        # Tratamento rigoroso de coordenadas
                        lat = self.safe_float(row.get('Latitude'))
                        lon = self.safe_float(row.get('Longitude'))
                        
                        # REGRA PRINCIPAL: Se coordenadas inválidas, estado deve ser UNKNOWN
                        if lat is None or lon is None or lat == 0.0 or lon == 0.0:
                            entry = {
                                'start_time': start_time,
                                'end_time': end_time,
                                'state': 'UNKNOWN',  # FORÇA UNKNOWN
                                'confidence': 0,  # Zera a confiança
                                'latitude': None,  # Dado faltante explícito
                                'longitude': None, # Dado faltante explícito
                                'duration': 0,
                                'tower_jump': False,
                                'same_time_diff_state': False,
                                'cell_types': row.get('CellType', '').lower().strip(),
                                'raw_data': row,
                                'conflict_resolution': 'INVALID_COORDS',
                                'discarded_state': None,
                                'resolved_by': 'COORD_VALIDATION',
                                'location_score': 0  # Score mínimo
                            }
                            data.append(entry)
                            continue
                        
                        # Só processa estado se tiver coordenadas válidas
                        state = self.precise_state_from_coordinates(lat, lon)
                        
                        # Fallbacks só para coordenadas válidas
                        if state == 'UNKNOWN':
                            location_text = ' '.join(filter(None, [
                                str(row.get('City', '')),
                                str(row.get('County', '')),
                                str(row.get('State', ''))
                            ])).strip()
                            state = self.extract_state(location_text)
                        
                        if state == 'UNKNOWN' and 'State' in row:
                            state_candidate = row['State'].strip().title()
                            if state_candidate in self.us_state_abbreviations.values():
                                state = state_candidate
                            elif len(state_candidate) == 2:
                                state = self.us_state_abbreviations.get(state_candidate.upper(), 'UNKNOWN')
                        
                        entry = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'state': state,
                            'confidence': 100,
                            'latitude': lat,
                            'longitude': lon,
                            'duration': 0,
                            'tower_jump': False,
                            'same_time_diff_state': False,
                            'cell_types': row.get('CellType', '').lower().strip(),
                            'raw_data': row,
                            'conflict_resolution': 'NO_CONFLICT',
                            'discarded_state': None,
                            'resolved_by': None,
                            'location_score': self.calculate_location_score({
                                'latitude': lat,
                                'longitude': lon,
                                'confidence': 100,
                                'cell_types': row.get('CellType', ''),
                                'state': state
                            })
                        }
                        data.append(entry)
                        
                    except Exception as e:
                        print(f"Erro na linha {row_num}: {str(e)}")
                        continue
                    
        except Exception as e:
            print(f"Erro ao ler arquivo: {str(e)}")
            return []

        # Pós-processamento
        data = sorted(data, key=lambda x: x['start_time'])
        
        # Verificação final de consistência
        invalid_coords = sum(1 for e in data if e['latitude'] is None or e['longitude'] is None)
        unknown_states = sum(1 for e in data if e['state'] == 'UNKNOWN')
        
        print(f"\nESTATÍSTICAS FINAIS:")
        print(f"Total de registros: {len(data)}")
        print(f"Registros com coordenadas inválidas: {invalid_coords} ({invalid_coords/len(data):.1%})")
        print(f"Registros com estado UNKNOWN: {unknown_states} ({unknown_states/len(data):.1%})")
        
        # Garante que todos os registros sem coordenadas têm estado UNKNOWN
        for entry in data:
            if entry['latitude'] is None or entry['longitude'] is None:
                entry['state'] = 'UNKNOWN'
        
        return data

    def generate_report(self, data, output_file):
        """Gera um relatório CSV com os dados processados"""
        if not data:
            print("Nenhum dado para gerar relatório")
            return False
        
        try:
            # Gera o relatório principal
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'start_time', 'end_time', 'state', 'confidence', 
                    'latitude', 'longitude', 'duration', 'tower_jump',
                    'same_time_diff_state', 'cell_types', 'location_score',
                    'conflict_resolution', 'discarded_state', 'resolved_by'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in data:
                    writer.writerow({
                        'start_time': entry['start_time'].strftime(self.date_format),
                        'end_time': entry['end_time'].strftime(self.date_format),
                        'state': entry['state'],
                        'confidence': entry['confidence'],
                        'latitude': entry['latitude'],
                        'longitude': entry['longitude'],
                        'duration': entry['duration'],
                        'tower_jump': entry['tower_jump'],
                        'same_time_diff_state': entry['same_time_diff_state'],
                        'cell_types': entry['cell_types'],
                        'location_score': entry['location_score'],
                        'conflict_resolution': entry['conflict_resolution'],
                        'discarded_state': entry['discarded_state'] or '',
                        'resolved_by': entry['resolved_by'] or ''
                    })

            return True
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            return False

    def calculate_state_stats(self, data):
        """Calcula estatísticas de distribuição por estado"""
        if not data:
            return {}
        
        state_counts = {}
        total = len(data)
        
        for entry in data:
            state = entry['state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Ordena por contagem decrescente
        sorted_stats = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Converte para porcentagens
        state_stats = {state: (count, count/total) for state, count in sorted_stats}
        
        return state_stats

    def run_analysis(self):
        """Método principal que executa toda a análise"""
        print("=== Tower Jump Analyzer ===")
        print("Sistema avançado de análise de localização e detecção de tower jumps\n")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../input")
        output_dir = os.path.join(base_dir, "../output")
        
        # Garante que as pastas existam
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = os.path.join(data_dir, "CarrierData.csv")
        output_file = os.path.join(output_dir, f"tower_jump_analysis_{timestamp}.csv")

        if not os.path.exists(input_file):
            print(f"Erro: Arquivo de entrada não encontrado em {input_file}")
            print("Certifique-se que o arquivo CarrierData.csv está na pasta 'localization'")
            return

        print("Carregando e processando dados...")
        data = self.load_data(input_file)

        if not data:
            print("Nenhum dado válido encontrado no arquivo de entrada.")
            return

        print("Corrigindo estados UNKNOWN consecutivos...")
        data = self.fix_consecutive_unknowns(data)

        # VERIFICAÇÃO FINAL - GARANTIR QUE REGISTROS SEM COORDENADAS SÃO UNKNOWN
        print("Verificando consistência final...")
        invalid_count = 0
        for entry in data:
            if (entry['latitude'] is None or entry['longitude'] is None) and entry['state'] != 'UNKNOWN':
                entry['state'] = 'UNKNOWN'
                entry['confidence'] = 0
                entry['location_score'] = 0
                entry['conflict_resolution'] = 'INVALID_COORDS'
                entry['resolved_by'] = 'FINAL_VALIDATION'
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"Corrigidos {invalid_count} registros inconsistentes na validação final")

        print("Detectando tower jumps...")
        data = self.detect_jumps(data)

        print("Resolvendo conflitos de estado...")
        data = self.resolve_state_conflicts(data)

        print("Obtendo localização atual...")
        current_location = self.get_current_location(data)

        print("Gerando relatório...")
        if self.generate_report(data, output_file):
            print(f"\nRelatório gerado com sucesso em: {output_file}")
        else:
            print("Falha ao gerar relatório.")
            return

        """Imprime um resumo dos dados processados"""
        if not data:
            print("Nenhum dado para resumo")
            return
        
        total_records = len(data)
        tower_jump_count = sum(1 for e in data if e['tower_jump'])
        unknown_state_count = sum(1 for e in data if e['state'] == 'UNKNOWN')
        
        print("\n=== RESUMO DOS DADOS ===")
        print(f"Total de registros: {total_records}")
        print(f"Registros com Tower Jump: {tower_jump_count} ({tower_jump_count/total_records:.2%})")
        print(f"Registros com estado UNKNOWN: {unknown_state_count} ({unknown_state_count/total_records:.2%})")
        
        # Adiciona estatísticas por estado
        state_stats = self.calculate_state_stats(data)
        if state_stats:
            print("\nDistribuição por estado:")
            print("{:<20} {:<10} {:<10}".format("Estado", "Contagem", "Porcentagem"))
            print("-" * 40)
            for state, (count, percentage) in state_stats.items():
                print("{:<20} {:<10} {:<10.2%}".format(state, count, percentage))

        print("\n=== LOCALIZAÇÃO ATUAL ===")
        if current_location['status'] == 'NO_DATA':
            print("Sem dados de localização")
        elif current_location['status'] == 'NO_VALID_DATA':
            print("Sem dados válidos de localização (todos descartados em conflitos)")
        else:
            print(f"Estado: {current_location['state']}")
            print(f"Status: {'Atual' if current_location['status'] == 'CURRENT' else 'Possivelmente desatualizada'}")
            print(f"Qualidade: {current_location['location_quality']} (Score: {current_location['location_score']}/100)")
            print(f"Última atualização: {current_location['last_update']} ({current_location['minutes_since_update']:.1f} minutos atrás)")
            print(f"Confiança: {current_location['confidence']}%")
            print(f"Tower Jump: {'Sim' if current_location['tower_jump'] else 'Não'}")
            print(f"Coordenadas: {current_location['coordinates']}")
            print(f"Tipo de célula: {current_location['cell_type'] or 'Desconhecido'}")

        print("\n=== RELATÓRIO GERADO ===")
        df = pd.read_csv(output_file)
        
        print(df)