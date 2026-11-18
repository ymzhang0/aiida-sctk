"""Retain response files verbatim: no undocumented column interpretation."""
from aiida import orm
from aiida.parsers import Parser


class SCTKParser(Parser):
    def parse(self, **kwargs):
        repository = self.retrieved.base.repository
        names = repository.list_object_names()
        stdout = self.node.get_option('output_filename')
        if stdout not in names:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
        text = repository.get_object_content(stdout)
        if '%%%%%%%%' in text or 'Error in routine' in text:
            return self.exit_codes.ERROR_SCTK_FAILED
        if 'JOB DONE.' not in text:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        if 'aiida.pi0_head.dat' not in names:
            return self.exit_codes.ERROR_POLARIZABILITY_MISSING
        for name, port in [('aiida.pi0_head.dat', 'polarizability'),
                           ('aiida.ixc_pi0_head.dat', 'ixc_polarizability')]:
            if name in names:
                with repository.open(name, 'rb') as handle:
                    self.out(port, orm.SinglefileData(file=handle, filename=name))
        self.out('output_parameters', orm.Dict(dict={
            'calculation': 'kel', 'completed': True,
            'response_files': [n for n in names if n.endswith('_head.dat')]}))
