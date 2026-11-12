# Headless PWML import for automated testing (added by the T2PW import harness).
#
#   rake "pwml:import[/path/file.pwml]"        one file
#   PWML_LIST=/path/list.txt rake pwml:import   one path per line
#
# Each file is imported with PwmlParser#parse (the same call pathways_controller#upload makes)
# on a CLEAN database: every data table except users/sessions/schema_migrations is truncated first.
# The last imported pathway is left in the DB so it can be viewed. Prints one line per file:
#   PWML_RESULT {json}
# Exit status: 0 if every file imported, 1 otherwise.
require 'json'
require 'stringio'

namespace :pwml do
  KEEP_TABLES = %w(users sessions schema_migrations visualization_templates drawable_images).freeze

  def pwml_tables
    ActiveRecord::Base.connection.tables - KEEP_TABLES
  end

  def pwml_counts
    c = ActiveRecord::Base.connection
    Hash[pwml_tables.map { |t| [t, c.select_value("SELECT COUNT(*) FROM `#{t}`").to_i] }]
  end

  def pwml_reset!
    c = ActiveRecord::Base.connection
    c.execute('SET FOREIGN_KEY_CHECKS=0')
    # only tables that hold rows: TRUNCATE on all ~100 tables costs ~20 s on this stack
    pwml_counts.each { |t, n| c.execute("TRUNCATE TABLE `#{t}`") if n > 0 }
    c.execute('SET FOREIGN_KEY_CHECKS=1')
    left = pwml_counts.select { |_, n| n > 0 }.keys
    raise "DB not clean after reset: #{left.inspect}" unless left.empty?
  end

  def pwml_plain(v)
    v = v.full_messages if v.respond_to?(:full_messages)
    v.is_a?(Hash) ? Hash[v.map { |k, x| [k.to_s, pwml_plain(x)] }] : (v.is_a?(Array) ? v.map { |x| pwml_plain(x) } : v.to_s)
  end

  # "imported" must mean something: group the row deltas into the objects a reader cares about
  MAIN = {
    'pathways' => %w(pathways),
    'compounds' => %w(compounds),
    'compound_locations' => %w(compound_locations),
    'proteins' => %w(proteins),
    'protein_complexes' => %w(protein_complexes),
    'reactions' => %w(reactions),
    'transports' => %w(transports),
    'element_states' => %w(element_states),
    'subcellular_locations' => %w(subcellular_locations),
    'species' => %w(species),
    'pathway_visualizations' => %w(pathway_visualizations)
  }.freeze

  # An empty DB has no species/compounds/etc, so the parser creates them, and every owned model
  # validates a user. Production never hits this because the elements already exist. The harness
  # owner is a local admin (also used to log in for screenshots); schema default user_id is 11 (a production guest), which does not exist here, so user_id is forced to the harness user on validation.
  HARNESS_EMAIL = 'admin@pwi.local'.freeze
  HARNESS_PASSWORD = 'pwi-local-only'.freeze

  def pwml_harness_user
    u = User.find_by(email: HARNESS_EMAIL) ||
        User.create!(email: HARNESS_EMAIL, password: HARNESS_PASSWORD, password_confirmation: HARNESS_PASSWORD,
                     role: 'admin', name: 'PWI Harness')
    unless $pwml_user_hook
      $pwml_user_hook = true
      uid = u.id
      ActiveRecord::Base.before_validation { self.user_id = uid if has_attribute?(:user_id); true }
    end
    u
  end

  # Tire (elasticsearch) indexes after every save; with no ES the save callback raises after the row
  # is written, which the parser swallows and then drops the id mapping. Stub the index writes only.
  def pwml_stub_search!
    return if $pwml_search_stubbed
    $pwml_search_stubbed = true
    Tire::Index.class_eval do
      %w(store update remove create delete refresh import).each { |m| define_method(m) { |*a, &b| {} } }
    end
  end

  def pwml_import_one(path)
    require 'pwml_parser'
    pwml_stub_search!
    pwml_harness_user
    row = { 'file' => path, 'success' => false, 'errors' => {}, 'pathway_id' => nil, 'created' => {}, 'created_main' => {} }
    t0 = Time.now
    out = StringIO.new
    real_stdout = $stdout
    begin
      pwml_reset!
      before = pwml_counts
      row['reset_seconds'] = (Time.now - t0).round(2)
      ENV['PWML_DEBUG'] = '1' # makes the parser report the exception it otherwise swallows
      $stdout = out
      result = PwmlParser.new.parse(File.open(path))
      $stdout = real_stdout
      row['success'] = !!result[:success]
      row['pathway_id'] = result[:pathway_id]
      row['pw_id'] = SuperPathwayVisualization.order(:id).last.try(:pw_id) if row['success']
      row['errors'] = pwml_plain(result[:errors] || {})
      after = pwml_counts
      row['created'] = Hash[after.map { |t, n| [t, n - before[t]] }.select { |_, d| d != 0 }]
      row['created_main'] = Hash[MAIN.map { |k, ts| [k, ts.map { |t| after[t].to_i - before[t].to_i }.inject(:+)] }]
    rescue Exception => e # parser can raise outside its own rescue (e.g. no super-pathway-visualization root)
      $stdout = real_stdout
      row['errors'] = { 'exception' => "#{e.class}: #{e.message}" }
    ensure
      $stdout = real_stdout
    end
    if (m = out.string.match(/EXCEPTION in parse\s+error_class: "([^"]*)"\s+error_message: "((?:[^"\\]|\\.)*)"/m))
      row['parser_exception'] = "#{m[1]}: #{m[2]}"
    end
    File.write(ENV['PWML_TRACE'], out.string) if ENV['PWML_TRACE'] # parser debug output, for diagnosis
    row['seconds'] = (Time.now - t0).round(2)
    row
  end

  desc 'Import PWML file(s) into a clean DB; prints PWML_RESULT json lines; exit 1 if any failed'
  task :import, [:path] => :environment do |_t, args|
    ActiveRecord::Base.logger = nil
    paths = args[:path] ? [args[:path]] : []
    paths += File.read(ENV['PWML_LIST']).split("\n").map(&:strip).reject(&:empty?) if ENV['PWML_LIST']
    abort 'usage: rake "pwml:import[path]" or PWML_LIST=file' if paths.empty?
    ok = true
    paths.each do |p|
      row = File.exist?(p) ? pwml_import_one(p) : { 'file' => p, 'success' => false, 'errors' => { 'exception' => 'file not found' } }
      ok &&= row['success']
      puts "PWML_RESULT #{JSON.generate(row)}"
      $stdout.flush
    end
    exit(ok ? 0 : 1)
  end
end
