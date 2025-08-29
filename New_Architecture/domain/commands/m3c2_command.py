# domain/commands/m3c2_command.py
class M3C2Command(Command):
    def __init__(self, runner: M3C2Runner):
        self.runner = runner
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mov = context['moving_cloud']
        ref = context['reference_cloud']
        params = context['m3c2_params']
        
        distances, uncertainties = self.runner.compute(
            mov, ref, params
        )
        
        context['distances'] = distances
        context['uncertainties'] = uncertainties
        return context