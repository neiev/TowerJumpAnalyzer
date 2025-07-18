import csv
import json
import os
import re
from datetime import datetime
import traceback
from shapely.geometry import Point, shape

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

        if entry['state'] == 'UNKNOWN':
            score -= 20


        # Penaliza se a localização for pontual (sem coordenadas)
        if entry['latitude'] is None or entry['longitude'] is None or \
            (entry['latitude'] == 0 and entry['longitude'] == 0):
            score -= 30

        if entry.get('tower_jump', False):
            score -= 50

        # Garante que o score não fique negativo
        score = max(score, 0)
        return min(score, 100)

        
   
    
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
    

    def calculate_velocity(self, time_diff, distance):
        """Calcula a velocidade entre dois pontos em km/h e a distância """
        if time_diff <= 0 or distance < 0:

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

        return  distance / (time_diff / 3600)  # km/h
    
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


        # previous['tower_jump'] = True
        # previous['discarded_state'] = previous['state'] 
        # previous['resolved_by'] = 'TOWER_JUMP_DETECTION'
        return current
    
    def is_valid_record(self, record):
        """Verifica se o registro é válido para análise"""
        if not record:
            return False
        
        if 'start_time' not in record or 'end_time' not in record:
            return False
        
        if not isinstance(record['start_time'], datetime) or not isinstance(record['end_time'], datetime):
            return False
        
        if record['start_time'] >= record['end_time']:
            return False
        
        if 'latitude' not in record or 'longitude' not in record:
            return False
        
        if record['latitude'] is None or record['longitude'] is None:
            return False
        
        if record['latitude'] == 0 and record['longitude'] == 0:
            return False
        
        if record['latitude'] <= -90 and record['longitude'] <= -180:
            return False

        return True
    
    def check_consistency(self, record, data, index):
        """Verifica se há registros consistentes subsequentes"""
        # Verificar próximos 3 registros
        consistent_count = 0
        for j in range(index + 1, min(index + 4, len(data))):
            next_rec = data[j]
            if not self.is_valid_record(next_rec):
                continue
                
            # Verificar mesma localização e estado
            same_location = self.haversine_distance(
                record['lat'], record['lon'],
                next_rec['lat'], next_rec['lon']
            ) < 1
            
            same_state = record['state'] == next_rec['state']
            
            if same_location and same_state:
                consistent_count += 1

        # Verificar 3 registros anteriores
        for j in range(index - 1, max(index - 4, -1), -1):
            prev_rec = data[j]
            if not self.is_valid_record(prev_rec):
                continue
                
            # Verificar mesma localização e estado
            same_location = self.haversine_distance(
                record['lat'], record['lon'],
                prev_rec['lat'], prev_rec['lon']
            ) < 1
            
            same_state = record['state'] == prev_rec['state']
            
            if same_location:
                consistent_count -= 2
            if same_state:
                consistent_count -= 2
                
        return consistent_count >= 2  # Pelo menos 2 registros consistentes

    def detect_jumps(self, data):
        """Detecta tower jumps com base nos dados de localização"""
        if not data:
            return data
        
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i - 1]
            
            if not current['start_time'] or not previous['start_time']:
                continue

            if not current['state'] or \
                current['state'] == '' or \
                current['state'] == 'UNKNOWN' or \
                current['state'] is None:
                # se não tem estado válido, cria um objeto padrão
                current = self.criar_objeto_padrao(current)
                continue

            if ((current['latitude'] is None or current['longitude'] is None or \
                current['latitude'] == 0 or current['longitude'] == 0) and \
                current['state'] == 'UNKNOWN'):
                current = self.criar_objeto_padrao(current)
                continue

            if 'tower_jump' not in previous:
                print(previous)
                previous['tower_jump'] = False

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
                previous['tower_jump'] = data[i]['tower_jump']

            if valid_previous_state == current['state'] and \
                current['state'] != 'UNKNOWN':
                if previous['start_time'] == current['start_time']:
                    current['same_time_diff_state'] = 'SAME_TIME_SAME_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFF_TIME_SAME_STATE'
            else:
                if previous['start_time'] == current['start_time']:
                    current['same_time_diff_state'] = 'SAME_TIME_DIFF_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFF_TIME_DIFF_STATE'


            time_diff = (current['start_time'] - previous['end_time']).total_seconds()

            current_point = Point(current['longitude'], current['latitude'])
            previous_point = Point(previous['longitude'], previous['latitude'])

            current['distance'] = previous_point.distance(current_point) * 111  # Distância em km

            current['velocity'] = self.calculate_velocity(
                current['distance'],
                time_diff
            )

            previous['velocity'] = previous.get('velocity', 0)
            current['vehicle_type'] = self.detect_vehicle_type(current['velocity'], current.get('distance', 0), time_diff)
        
            current['location_score'] = self.calculate_location_score(current)
            previous['location_score'] = self.calculate_location_score(previous)

            current['duration'] = (current['start_time'] - previous['end_time']).total_seconds() / 60

            # Verifica se a velocidade é muito alta
            if current['velocity'] > self.max_velocity_threshold:
                current['conflict_resolution'] = 'HIGH_VELOCITY'
                self.marcar_tower_jump(current, previous)
                continue

            is_same_state = current['state'] == previous['state']

            if is_same_state and \
                current['same_time_diff_state'] == 'SAME_TIME_SAME_STATE':
                current['conflict_resolution'] = 'SAME_STATE_SAME_TIME'
                if previous['tower_jump']:
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue

            elif is_same_state and \
                current['same_time_diff_state'] == 'DIFF_TIME_SAME_STATE':
                current['conflict_resolution'] = 'SAME_STATE_DIFF_TIME'
                if previous['tower_jump'] and time_diff < self.min_time_diff_threshold:
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue
            elif not is_same_state and \
                current['same_time_diff_state'] == 'SAME_TIME_DIFF_STATE':
                current['conflict_resolution'] = 'DIFF_STATE_SAME_TIME'
                # checa o estado válido para o intervalo de tempo de registro
                is_consistent = self.check_consistency(current, data, i)    
                if not is_consistent:
                    current['conflict_resolution'] = 'DIFF_STATE_JUMP'
                    current = self.marcar_tower_jump(current, valid_previous_state)
                    continue


            # Verifica se o deslocamento é possível
            is_location_change_possible = False
            # calcule com base na distância e tempo e velocidade e tipo de veículo
            # não usa apenas o minimo e máximo e sim faz cáluclo matemático real de deslocamento
            if current['distance'] > 0 and time_diff > 0:
                is_location_change_possible = (current['velocity'] * (time_diff / 3600)) >= current['distance']
            current['is_location_change_possible'] = is_location_change_possible

            # Verifica se a velocidade é impossível
            if (not is_location_change_possible):
                current['conflict_resolution'] = 'LOCATION_CHANGE_IMPOSSIBLE'
                current = self.marcar_tower_jump(current, valid_previous_state)
                continue

            # 1. Registros simultâneos em estados diferentes:
            if (not is_same_state and 
                not current['same_time_diff_state'] == 'SAME_TIME_DIFF_STATE'):
                current['conflict_resolution'] = 'SIMULTANEOUS_STATE_JUMP'
                # checa o estado válido para o intervalo de tempo de registro
                is_consistent = self.check_consistency(current, data, i)
                if not is_consistent:
                    current['conflict_resolution'] = 'DIFF_STATE_JUMP'
                    current = self.marcar_tower_jump(current, valid_previous_state)
                continue
            # 2. Registros com estado diferente e tempo de diferença curto
            elif (not is_same_state and 
                  current['same_time_diff_state'] == 'DIFF_TIME_DIFF_STATE' and 
                  time_diff < self.min_time_diff_threshold):
                current['conflict_resolution'] = 'DIFF_STATE_SHORT_TIME'
                # checa o estado válido para o intervalo de tempo de registro
                is_consistent = self.check_consistency(current, data, i)
                if not is_consistent and \
                   not current['is_location_change_possible']:
                    current = self.marcar_tower_jump(current, valid_previous_state)
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
        """Corrige sequências de estados UNKNOWN com base nos vizinhos conhecidos"""
        if not data:
            return data
        
        for i in range(len(data)):
            if data[i]['state'] == 'UNKNOWN':
                next_known = None
                for j in range(i+1, len(data)):
                    if data[j]['state'] != 'UNKNOWN':
                        next_known = data[j]['state']
                        break
                
                prev_known = None
                for j in range(i-1, -1, -1):
                    if data[j]['state'] != 'UNKNOWN':
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
        """Carrega e processa os dados do arquivo CSV"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                try:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                    has_header = csv.Sniffer().has_header(f.read(1024))
                    f.seek(0)
                except:
                    dialect = 'excel'
                    has_header = True
                
                reader = csv.DictReader(f, dialect=dialect)
                
                required_columns = {'UTCDateTime', 'LocalDateTime', 'State', 'Latitude', 'Longitude'}
                if not all(col in reader.fieldnames for col in required_columns):
                    missing = required_columns - set(reader.fieldnames)
                    print(f"Erro: Colunas obrigatórias faltando: {missing}")
                    return []
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        datetime_str = row.get('UTCDateTime') or row.get('LocalDateTime')
                        start_time = self.parse_datetime(datetime_str)
                        
                        if not start_time:
                            print(f"Aviso: Ignorando linha {row_num} - formato de data/hora inválido: {datetime_str}")
                            continue

                        end_time = start_time
                        
                        lat = self.safe_float(row.get('Latitude'))
                        lon = self.safe_float(row.get('Longitude'))
                        
                        state = self.precise_state_from_coordinates(lat, lon)
                        
                        if state == 'UNKNOWN':
                            location_text = ' '.join([
                                str(row.get('City', '')),
                                str(row.get('County', '')),
                                str(row.get('State', ''))
                            ]).strip()
                            state = self.extract_state(location_text)
                        
                        if state == 'UNKNOWN' and 'State' in row:
                            state_candidate = row['State'].strip().title()
                            if state_candidate in self.us_state_abbreviations.values():
                                state = state_candidate
                            elif len(state_candidate) == 2:
                                state = self.us_state_abbreviations.get(state_candidate.upper(), 'UNKNOWN')
                        
                        if state == 'UNKNOWN' and lat is not None and lon is not None:
                            state = self.infer_state_from_coordinates(lat, lon)
                        
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
                            'location_score': 0,
                            'velocity': 0,
                            'vehicle_type': 'UNKNOWN'
                        }
                        
                        entry['location_score'] = self.calculate_location_score(entry)
                        data.append(entry)
                        
                    except Exception as e:
                        print(f"Erro ao processar linha {row_num}: {str(e)}")
                        continue
                    
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {file_path}")
            return []
        except Exception as e:
            print(f"Erro inesperado ao ler arquivo: {str(e)}")
            return []

        if not data:
            print("Aviso: O arquivo foi lido, mas nenhum dado válido foi encontrado")
            return []

        data = sorted(data, key=lambda x: x['start_time'])
        print(f"Carregados {len(data)} registros válidos")
        
        unknown_count = sum(1 for e in data if e['state'] == 'UNKNOWN')
        if unknown_count > 0:
            print(f"AVISO: {unknown_count} registros ({unknown_count/len(data):.2%}) com estado UNKNOWN")
            if unknown_count/len(data) > 0.05:
                print("  Ação recomendada: Verificar qualidade das coordenadas e mapeamento de estados")
        
        return data

    # def generate_report(self, data, output_file):
        """Gera um relatório CSV com os dados processados"""
        if not data:
            print("Nenhum dado para gerar relatório")
            return False
        
        try:
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
        
    # def print_summary(self, data):
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
            
            # Gera um arquivo adicional com estatísticas por estado
            stats_file = os.path.join(os.path.dirname(output_file), "StateStatistics.csv")
            state_stats = self.calculate_state_stats(data)
            
            with open(stats_file, 'w', newline='', encoding='utf-8') as sf:
                writer = csv.writer(sf)
                writer.writerow(["State", "Count", "Percentage"])
                for state, (count, percentage) in state_stats.items():
                    writer.writerow([state, count, f"{percentage:.2%}"])
            
            return True
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            traceback.print_exc()
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

    def print_summary(self, data):
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

def main():
    print("=== Tower Jump Analyzer ===")
    print("Sistema avançado de análise de localização e detecção de tower jumps\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "localization")
    input_file = os.path.join(data_dir, "CarrierData.csv")
    output_file_name = 'main_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    output_file = os.path.join(base_dir, output_file_name)


    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada não encontrado em {input_file}")
        print("Certifique-se que o arquivo CarrierData.csv está na pasta 'localization'")
        return

    analyzer = TowerJumpAnalyzer()
    print("Carregando e processando dados...")
    data = analyzer.load_data(input_file)

    if not data:
        print("Nenhum dado válido encontrado no arquivo de entrada.")
        return

    print("Corrigindo estados UNKNOWN consecutivos...")
    data = analyzer.fix_consecutive_unknowns(data)

    print("Detectando tower jumps...")
    data = analyzer.detect_jumps(data)

    print("Resolvendo conflitos de estado...")
    data = analyzer.resolve_state_conflicts(data)

    print("Obtendo localização atual...")
    current_location = analyzer.get_current_location(data)

    print("Gerando relatório...")
    if analyzer.generate_report(data, output_file):
        print(f"\nRelatório gerado com sucesso em: {output_file}")
    else:
        print("Falha ao gerar relatório.")
        return

    analyzer.print_summary(data)

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

if __name__ == "__main__":
    main()
