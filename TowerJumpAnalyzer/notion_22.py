from datetime import datetime
import math
import csv
import os 


class TowerJumpAnalyzer:
    def __init__(self, config=None):
        # Configurações padrão
        self.max_velocity_threshold = 250  # km/h - velocidade máxima realista
        self.min_time_diff_threshold = 300  # segundos (5 minutos)
        self.max_confidence_threshold = 95  # percentual máximo de confiança
        self.min_confidence_threshold = 50  # percentual mínimo de confiança
        self.date_format = '%m/%d/%y %H:%M'
        
        # Sobrescreve com configurações personalizadas
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
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
                        
                        # Determina o estado - prioridade para coordenadas
                        state = 'UNKNOWN'
                        
                        if lat is not None and lon is not None:
                            state = row.get('State', '')
                        
                        # Cria a entrada de dados
                        entry = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'state': state,
                            'confidence': 100,  # Assume alta confiança para dados com coordenadas
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
                        
                        # entry['location_score'] = self.calculate_location_score(entry)
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

    
    def detect_jumps(self, data):
        """
        Método principal para detecção de tower jumps
        Implementa uma abordagem mais abrangente para detectar todos os tipos possíveis
        """
        if not data or len(data) < 2:
            return data
            
        # Pré-processamento: calcula distâncias, velocidades e categoriza registros
        data = self.preprocess_data(data)
        
        # Aplica algoritmo de detecção de tower jumps
        data = self.apply_detection_logic(data)
        
        # Pós-processamento: refina resultados
        data = self.postprocess_results(data)
        
        return data
    
    def preprocess_data(self, data):
        """Prepara os dados para análise"""
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i-1]
            
            # Calcula distância entre pontos (em km)
            current['distance'] = self.calculate_distance(
                previous['latitude'], previous['longitude'],
                current['latitude'], current['longitude']
            ) if all(p is not None for p in [
                previous['latitude'], previous['longitude'],
                current['latitude'], current['longitude']
            ]) else 0
            
            # Calcula tempo entre registros (em segundos)
            time_diff = (current['start_time'] - previous['end_time']).total_seconds()
            current['time_diff'] = time_diff
            
            # Calcula velocidade (km/h)
            if time_diff > 0:
                current['velocity'] = (current['distance'] / time_diff) * 3600
            else:
                current['velocity'] = 0
                
            # Classifica o tipo de veículo com base na velocidade
            current['vehicle_type'] = self.classify_vehicle_type(current['velocity'])
            
            # Classifica a relação tempo-estado entre registros
            if time_diff < 1:  # praticamente simultâneo
                if current['state'] == previous['state']:
                    current['same_time_diff_state'] = 'SAME_TIME_SAME_STATE'
                else:
                    current['same_time_diff_state'] = 'SAME_TIME_DIFF_STATE'
            else:
                if current['state'] == previous['state']:
                    current['same_time_diff_state'] = 'DIFF_TIME_SAME_STATE'
                else:
                    current['same_time_diff_state'] = 'DIFF_TIME_DIFF_STATE'
            
            # Inicializa campos de análise
            current['is_tower_jump'] = False
            current['discarded_state'] = None
            current['resolved_by'] = None
            current['conflict_resolution'] = 'NO_CONFLICT'
            
        return data
    
    def apply_detection_logic(self, data):
        """Aplica lógica de detecção abrangente para Tower Jumps"""
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i-1]
            time_diff = current['time_diff']
            
            # CASO 1: Velocidade impossível
            if current['velocity'] > self.max_velocity_threshold:
                self.mark_tower_jump(current, previous, 'HIGH_VELOCITY')
                continue
                
            # CASO 2: Registros simultâneos em estados diferentes
            if current['same_time_diff_state'] == 'SAME_TIME_DIFF_STATE':
                # Verificação adicional de consistência com registros vizinhos
                if not self.check_state_consistency(data, i):
                    self.mark_tower_jump(current, previous, 'SIMULTANEOUS_STATE_CONFLICT')
                continue
                
            # CASO 3: Mudança de estado em tempo insuficiente
            if (current['same_time_diff_state'] == 'DIFF_TIME_DIFF_STATE' and 
                time_diff < self.min_time_diff_threshold):
                
                # Verifica se o deslocamento é fisicamente possível
                if not self.is_movement_physically_possible(current):
                    self.mark_tower_jump(current, previous, 'IMPOSSIBLE_STATE_CHANGE')
                # Verifica consistência com registros anteriores e posteriores
                elif not self.check_state_consistency(data, i):
                    self.mark_tower_jump(current, previous, 'INCONSISTENT_STATE_CHANGE')
                continue
                
            # CASO 4: Padrão de oscilação entre estados (ping-pong)
            if i >= 2 and current['state'] == data[i-2]['state'] and current['state'] != previous['state']:
                self.mark_tower_jump(previous, data[i-2], 'PING_PONG_PATTERN')
                continue
                
            # CASO 5: Alta velocidade seguida de baixa velocidade com mudança de estado
            if (i >= 2 and current['velocity'] < 5 and previous['velocity'] > 100 and
                current['state'] != previous['state']):
                self.mark_tower_jump(current, previous, 'VELOCITY_ANOMALY')
                continue
                
            # CASO 6: Mudança de estado sem mudança significativa de coordenadas
            if (current['state'] != previous['state'] and 
                current['distance'] < 0.1 and time_diff > 60):
                self.mark_tower_jump(current, previous, 'STATIONARY_STATE_CHANGE')
                continue
                
            # CASO 7: Conflito com dados de tipo de célula
            if (current['state'] != previous['state'] and
                self.has_cell_type_conflict(current, previous)):
                self.mark_tower_jump(current, previous, 'CELL_TYPE_CONFLICT')
                continue
        
        return data
    
    def mark_tower_jump(self, current, reference, reason):
        """Marca um registro como tower jump e armazena informações relevantes"""
        current['is_tower_jump'] = True
        current['conflict_resolution'] = reason
        current['discarded_state'] = current['state']
        current['state'] = reference['state']  # Adota o estado de referência
        current['resolved_by'] = 'ALGORITHM'
        return current
    
    def check_state_consistency(self, data, index):
        """Verifica consistência do estado atual com registros vizinhos"""
        current = data[index]
        
        # Verifica registros anteriores (até 3, se disponíveis)
        prior_states = []
        for i in range(1, 4):
            if index - i >= 0:
                prior_states.append(data[index - i]['state'])
                
        # Verifica registros posteriores (até 3, se disponíveis)
        next_states = []
        for i in range(1, 4):
            if index + i < len(data):
                next_states.append(data[index + i]['state'])
                
        # Determina o estado mais provável com base na frequência
        all_states = prior_states + next_states
        if not all_states:
            return True  # Sem contexto suficiente
            
        state_counts = {}
        for state in all_states:
            state_counts[state] = state_counts.get(state, 0) + 1
            
        most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
        
        # Se o estado atual não corresponde ao mais comum e o mais comum é consistente
        return current['state'] == most_common_state or state_counts[most_common_state] <= len(all_states) / 2
    
    def is_movement_physically_possible(self, current):
        """Verifica se o deslocamento é fisicamente possível"""
        # Considere o tipo de veículo e velocidade máxima realista
        max_possible_distance = (current['velocity'] * (current['time_diff'] / 3600)) * 1.1  # 10% de margem
        return current['distance'] <= max_possible_distance
    
    def has_cell_type_conflict(self, current, previous):
        """Verifica conflitos baseados no tipo de célula de telefonia"""
        # Implementação simplificada - expandir conforme necessário
        if not current['cell_types'] or not previous['cell_types']:
            return False
            
        # Se há mudança de 4G para 2G com mudança de estado, pode indicar tower jump
        return ('4g' in previous['cell_types'] and '2g' in current['cell_types'] and
                current['state'] != previous['state'])
    
    def postprocess_results(self, data):
        """Refina os resultados após a detecção inicial"""
        # Identifica e corrige clusters de tower jumps
        for i in range(1, len(data) - 1):
            if (data[i]['is_tower_jump'] and not data[i-1]['is_tower_jump'] and 
                not data[i+1]['is_tower_jump'] and 
                data[i-1]['state'] == data[i+1]['state']):
                
                # Tower jump isolado entre estados consistentes
                data[i]['resolved_by'] = 'ISOLATED_CORRECTION'
                
        # Marca áreas fronteiriças para monitoramento especial
        for i in range(1, len(data) - 5):
            if self.is_border_area(data[i:i+5]):
                for j in range(i, i+5):
                    if not data[j]['is_tower_jump']:
                        data[j]['conflict_resolution'] = 'BORDER_AREA_CAUTION'
                
        return data
    
    def is_border_area(self, data_segment):
        """Identifica áreas de fronteira com base em padrões de mudança de estado"""
        states = [entry['state'] for entry in data_segment]
        unique_states = set(states)
        
        if len(unique_states) < 2:
            return False
            
        # Conta transições entre estados
        transitions = sum(1 for i in range(len(states) - 1) if states[i] != states[i+1])
        
        # Áreas de fronteira tendem a ter múltiplas transições em curto período
        return transitions >= 2
    
    def classify_vehicle_type(self, velocity):
        """Classifica o tipo de veículo com base na velocidade"""
        if velocity <= 5:
            return 'STATIONARY'
        elif velocity <= 10:
            return 'WALKING'
        elif velocity <= 25:
            return 'CYCLING'
        elif velocity <= 120:
            return 'DRIVING'
        elif velocity <= 300:
            return 'HIGH_SPEED_TRAIN'
        else:
            return 'AIRCRAFT'
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcula a distância entre dois pontos usando a fórmula de Haversine"""
        # Implementação da fórmula de Haversine
        R = 6371  # Raio da Terra em km
        
        # Converte coordenadas de graus para radianos
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Diferença entre latitudes e longitudes
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Fórmula de Haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
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
    # Outros métodos como parse_datetime, safe_float, etc. permanecem inalterados

    def generate_report(self, data, output_file):
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
                    'velocity', 'conflict_resolution', 'discarded_state', 
                    'resolved_by', 'vehicle_type'
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
                        'velocity': entry['velocity'],
                        'conflict_resolution': entry['conflict_resolution'],
                        'discarded_state': entry['discarded_state'] or '',
                        'resolved_by': entry['resolved_by'] or '',
                        'vehicle_type': entry['vehicle_type'] or 'UNKNOWN'
                    })
            return True
        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            return False
        
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

    print("Detectando tower jumps...")
    data = analyzer.detect_jumps(data)

    # print("Resolvendo conflitos de estado...")
    # data = analyzer.resolve_state_conflicts(data)

    # print("Obtendo localização atual...")
    # current_location = analyzer.get_current_location(data)

    print("Gerando relatório...")
    if analyzer.generate_report(data, output_file):
        print(f"\nRelatório gerado com sucesso em: {output_file}")
    else:
        print("Falha ao gerar relatório.")
        return

    analyzer.print_summary(data)

if __name__ == "__main__":
    main()