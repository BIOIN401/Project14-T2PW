# This class is for parsing PWML
class PwmlParser

  def initialize()
    # Hashes to stores all the elements for this visualizations
    # This is because pwml may contain elements that are not in PathWhiz, thus we assume
    # that all ids numbers are internal to the PWML document
    @cell_type_list = {}
    @species_list = {}
    @subcellular_location_list = {}
    @tissue_list = {}
    @biological_state_list = {}

    @edge_list = {}

    @bound_list = {}
    @bound_visualization_list = {}
    @compound_list = {}
    @compound_location_list = {}
    @element_collection_list = {}
    @element_collection_location_list = {}
    @nucleic_acid_list = {}
    @nucleic_acid_location_list = {}
    @protein_list = {}
    @protein_location_list = {}
    @protein_complex_list = {}
    @protein_complex_visualization_list = {}
    
    @reaction_list = {}
    @reaction_coupled_transport_list = {}
    @transport_list = {}
    @interaction_list = {}
    @sub_pathway_list = {}

    @result = {}
    @result[:errors] = {}
  end
 
  def parse(file)
    doc = Nokogiri::XML(file) do |config|
      config.strict.noblanks
    end
    super_pathway_visualization = SuperPathwayVisualization.new()
    # Find super pathway visualization node
    if doc.xpath('super-pathway-visualization').present?

      # Encapsulate this all in a rescue block so if there are errors, we can remove the newly created elements
      begin

        # Get named for information
        if doc.xpath('super-pathway-visualization/named-for-type').present? && doc.xpath('super-pathway-visualization/named-for-id').present?
          super_pathway_visualization.named_for_type = doc.xpath('super-pathway-visualization/named-for-type').first.content
          # Remember this id, we have to find it in the document
          named_for_doc_id = doc.xpath('super-pathway-visualization/named-for-id').first.content
        end

        # Get each pathway visualization context
        doc.xpath('super-pathway-visualization/pathway-visualization-contexts/pathway-visualization-context').each do |pvc_node|

          # Create new pathway visualization context and set the position
          pathway_visualization_context = PathwayVisualizationContext.new(to_hash_shallow(pvc_node))

          # Reset these for each pathway visualization
          # We store them OUTSIDE the pathway visualization loops because we need the elements for
          # the context, if it's present
          @cell_type_list = {}
          @species_list = {}
          @subcellular_location_list = {}
          @tissue_list = {}
          @biological_state_list = {}

          @edge_list = {}

          @bound_list = {}
          @bound_visualization_list = {}
          @compound_list = {}
          @compound_location_list = {}
          @element_collection_list = {}
          @element_collection_location_list = {}
          @nucleic_acid_list = {}
          @nucleic_acid_location_list = {}
          @protein_list = {}
          @protein_location_list = {}
          @protein_complex_list = {}
          @protein_complex_visualization_list = {}
          
          @reaction_list = {}
          @reaction_coupled_transport_list = {}
          @transport_list = {}
          @interaction_list = {}
          @sub_pathway_list = {}
          
          if pvc_node.xpath('pathway-visualization').present?
            pv_node = pvc_node.xpath('pathway-visualization').first
            pathway_visualization = PathwayVisualization.new(self.to_hash_shallow(pv_node))

            # Store hashes of all cell types, species, subcellular locations, and biological states
            # We do these first since we need the species for the pathway
            pv_node.xpath('cell-types/cell-type').each do |ct_node|
              self.store_object(ct_node)
            end
            pv_node.xpath('species/species').each do |o_node|
              self.store_object(o_node)
            end
            pv_node.xpath('subcellular-locations/subcellular-location').each do |sl_node|
              self.store_object(sl_node)
            end
            pv_node.xpath('tissues/tissue').each do |t_node|
              self.store_object(t_node)
            end
            pv_node.xpath('biological-states/biological-state').each do |bs_node|
              self.store_object(bs_node)
            end

            # Then get the pathway
            if pv_node.xpath('pathway').present?
              p_node = pv_node.xpath('pathway').first
              pathway = Pathway.new(self.to_hash_shallow(p_node))
              # If the name already exists, we give it a new one so the import can proceed
              if Pathway.where(name: pathway.name).count > 0
                pathway.name = pathway.name + " " + Time.now.to_i.to_s
              end
              # Get the sub pathways
              if p_node.xpath('sub-pathways').present?
                p_node.xpath('sub-pathways/sub-pathway').each do |sp_node|
                  self.store_nested_object(sp_node, pathway)
                end
              end
              # Get the references
              if p_node.xpath('references').present?
                p_node.xpath('references/reference').each do |r_node|
                  reference = get_child_object(r_node, pathway)
                  if reference.present?
                    pathway.references << reference
                  end
                end
              end

              # If the pathway saves
              if pathway.save
                # Check if this is the named_for element for the super pathway visualization, if so set it
                if super_pathway_visualization.named_for_type == "Pathway" && p_node.xpath('id').first.try(:content) == named_for_doc_id
                  super_pathway_visualization.named_for_id = pathway.id
                end

                pathway_visualization.pathway_id = pathway.id
              else
                id = p_node.xpath('id').first.try(:content)
                @result[:errors]["pathway_#{id}".to_sym] = pathway.errors.full_messages
              end
            end

            # Store hashes of all compounds, element collections, nucleic acids, proteins, bounds, protein_complexes
            pv_node.xpath('compounds/compound').each do |c_node|
             self.store_object(c_node)
            end
            pv_node.xpath('element-collections/element-collection').each do |ec_node|
              self.store_object(ec_node)
            end
            pv_node.xpath('nucleic-acids/nucleic-acid').each do |na_node|
              self.store_object(na_node)
            end
            pv_node.xpath('proteins/protein').each do |p_node|
              self.store_object(p_node)
            end
            pv_node.xpath('bounds/bound').each do |b_node|
              self.store_nested_object(b_node)
            end
            pv_node.xpath('protein-complexes/protein-complex').each do |p_node|
              self.store_nested_object(p_node)
            end

            # Store hashes of all reaction, reaction-coupled transports, transport, interactions
            pv_node.xpath('reactions/reaction').each do |r_node|
              self.store_nested_object(r_node)
            end
            pv_node.xpath('reaction-coupled-transports/reaction-coupled-transport').each do |r_node|
              self.store_nested_object(r_node)
            end
            pv_node.xpath('transports/transport').each do |t_node|
              self.store_nested_object(t_node)
            end
            pv_node.xpath('interactions/interaction').each do |i_node|
              self.store_nested_object(i_node)
            end

            if pathway_visualization.save
              # Store hashes of all compound locations, element collection locations, nucleic acid locations,
              # protein locations, protein_complex visualizations, and bound visualizations
              pv_node.xpath('compound-locations/compound-location').each do |cl_node|
                self.store_nested_visualization(cl_node, pathway_visualization)
              end
              pv_node.xpath('element-collection-locations/element-collection-location').each do |ec_node|
                self.store_nested_visualization(ec_node, pathway_visualization)
              end
              pv_node.xpath('nucleic-acid-locations/nucleic-acid-location').each do |na_node|
                self.store_nested_visualization(na_node, pathway_visualization)
              end
              pv_node.xpath('protein-locations/protein-location').each do |p_node|
                self.store_nested_visualization(p_node, pathway_visualization)
              end
              pv_node.xpath('edges/edge').each do |e_node|
                self.store_nested_visualization(e_node, pathway_visualization)
              end
              pv_node.xpath('bound-visualizations/bound-visualization').each do |b_node|
                self.store_nested_visualization(b_node, pathway_visualization)
              end
              pv_node.xpath('protein-complex-visualizations/protein-complex-visualization').each do |p_node|
                self.store_nested_visualization(p_node, pathway_visualization)
              end

              # Convert reaction, reaction coupled transport, transport, interaction, and sub-pathway visualizations
              pv_node.xpath('reaction-visualizations/reaction-visualization').each do |r_node|
                v = self.build_nested_visualization(r_node, pathway_visualization)
                if v.present?
                  pathway_visualization.reaction_visualizations << v
                end
              end
              pv_node.xpath('reaction-coupled-transport-visualizations/reaction-coupled-transport-visualization').each do |r_node|
                v = self.build_nested_visualization(r_node, pathway_visualization)
                if v.present?
                  pathway_visualization.reaction_coupled_transport_visualizations << v
                end
              end
              pv_node.xpath('transport-visualizations/transport-visualization').each do |t_node|
                v = self.build_nested_visualization(t_node, pathway_visualization)
                if v.present?
                  pathway_visualization.transport_visualizations << v
                end
              end
              pv_node.xpath('interaction-visualizations/interaction-visualization').each do |i_node|
                v = self.build_nested_visualization(i_node, pathway_visualization)
                if v.present?
                  pathway_visualization.interaction_visualizations << v
                end
              end
              pv_node.xpath('sub-pathway-visualizations/sub-pathway-visualization').each do |sp_node|
                v = self.build_nested_visualization(sp_node, pathway_visualization)
                if v.present?
                  pathway_visualization.sub_pathway_visualizations << v
                end
              end

              # Convert vacuous visualizations
              pv_node.xpath('vacuous-compound-visualizations/vacuous-compound-visualization').each do |vcv_node|
                v = self.build_nested_visualization(vcv_node, pathway_visualization)
                if v.present?
                  pathway_visualization.vacuous_compound_visualizations << v
                end
              end
              pv_node.xpath('vacuous-edge-visualizations/vacuous-edge-visualization').each do |vev_node|
                v = self.build_nested_visualization(vcv_node, pathway_visualization)
                if v.present?
                  pathway_visualization.vacuous_edge_visualizations << v
                end
              end
              pv_node.xpath('vacuous-element-collection-visualizations/vacuous-element-collection-visualization').each do |vecv_node|
                v = self.build_nested_visualization(vecv_node, pathway_visualization)
                if v.present?
                  pathway_visualization.vacuous_element_collection_visualizations << v
                end
              end
              pv_node.xpath('vacuous-nucleic-acid-visualizations/vacuous-nucleic-acid-visualization').each do |vnav_node|
                v = self.build_nested_visualization(vnav_node, pathway_visualization)
                if v.present?
                  pathway_visualization.vacuous_nucleic_acid_visualizations << v
                end
              end
              pv_node.xpath('vacuous-protein-visualizations/vacuous-protein-visualization').each do |vpv_node|
                v = self.build_nested_visualization(vpv_node, pathway_visualization)
                if v.present?
                  pathway_visualization.vacuous_protein_visualizations << v
                end
              end

              # Convert drawable element locations, membrane visualizations, label locations, and zoom visualizations
              pv_node.xpath('drawable-element-locations/drawable-element-location').each do |del_node|
                v = self.build_nested_visualization(del_node, pathway_visualization)
                if v.present?
                  pathway_visualization.drawable_element_locations << v
                end
              end
              pv_node.xpath('membrane-visualizations/membrane-visualization').each do |mv_node|
                v = self.build_nested_visualization(mv_node, pathway_visualization)
                if v.present?
                  pathway_visualization.membrane_visualizations << v
                end
              end
              pv_node.xpath('label-locations/label-location').each do |ll_node|
                v = self.build_nested_visualization(ll_node, pathway_visualization)
                if v.present?
                  pathway_visualization.label_locations << v
                end
              end
              pv_node.xpath('zoom-visualizations/zoom-visualization').each do |zv_node|
                v = self.build_nested_visualization(zv_node, pathway_visualization)
                if v.present?
                  pathway_visualization.zoom_visualizations << v
                end
              end

              # Set pathway visualization context pathway visualization
              pathway_visualization_context.pathway_visualization = pathway_visualization
            else
              # The visualization didn't save, so destroy the associated pathway, if it's present
              pathway_visualization.pathway.try(:destroy)
              id = pv_node.xpath('id').first.try(:content)
              @result[:errors]["pathway_visualization_#{id}".to_sym] = pathway_visualization.errors.full_messages
            end
          end

          if pvc_node.xpath('context').present?
            c_node = pvc_node.xpath('context').first
            context = Context.new(self.to_hash_deep(c_node))

            # Get the references
            if c_node.xpath('references').present?
              c_node.xpath('references/reference').each do |r_node|
                reference = get_child_object(r_node, pathway)
                if reference.present?
                  object.references << reference
                end
              end
            end

            # If the context saves
            if context.save
              # Check if this is the named_for element for the super pathway visualization, if so set it
              if super_pathway_visualization.named_for_type == "Context" && c_node.xpath('id').first.try(:conten) == named_for_doc_id
                super_pathway_visualization.named_for_id = context.id
              end

              pathway_visualization_context.context_id = context.id
            else
              id = c_node.xpath('id').first.try(:content)
              @result[:errors]["context_#{id}".to_sym] = context.errors.full_messages
            end
          end

          # Add the pathway visualization context to the super pathway visualization
          super_pathway_visualization.pathway_visualization_contexts << pathway_visualization_context
        end

      rescue
        super_pathway_visualization.pathway_visualization_contexts.each do |pvc|
          pvc.context.try(:destroy)
          pvc.pathway_visualization.pathway.try(:destroy)
          pvc.pathway_visualization.try(:destroy)
        end
        @result[:success] = false
        @result[:errors][:base] = "Error parsing PWML, please check your input and try again or contact and administrator"
        return @result
      end
    end

    if super_pathway_visualization.save
      # If the super pathway visualization saves, update the primary_spv for each of it's pathways
      super_pathway_visualization.pathway_visualization_contexts.each do |pvc|
        pvc.pathway_visualization.pathway.update_attribute(:primary_spv, super_pathway_visualization.id)
      end
      @result[:success] = true
      @result[:pathway_id] = super_pathway_visualization.pathway_visualization_contexts.first.pathway_visualization.pathway.id
    else
      # Destroy the created contexts and pathways
      super_pathway_visualization.pathway_visualization_contexts.each do |pvc|
        pvc.context.try(:destroy)
        pvc.pathway_visualization.pathway.try(:destroy)
        pvc.pathway_visualization.try(:destroy)
      end
      @result[:success] = false
      @result[:errors][:super_pathway_visualization] = super_pathway_visualization.errors
    end
    return @result
  end

  # Converts xml to a hash, only taking attributes on the first level, this omits any nested associations
  def to_hash_shallow(node)
    hash = {}
    klass = node.name.underscore.camelize.constantize
    node.element_children.each do |e|
      if e.element_children.length < 1
        hash = add_content(e, hash)
      end
    end
    return hash.slice(*attribute_list(klass))
  end

  # Converts xml to a hash recursively, for assocations of all levels
  def to_hash_deep(node)
    hash = {}
    return to_hash_recursive(node, hash)
  end

  def to_hash_recursive(node, hash)
    # Get the class
    klass = node.name.underscore.camelize.constantize
    node.element_children.each do |e|
      if e.element_children.length > 1
        # If there are more than one children create an array and store the hashed objects in the array
        nested_hash = {}
        if hash[e.name.underscore.to_sym] == nil
          hash[e.name.underscore.to_sym] = []
        end
        hash[e.name.underscore.to_sym] << to_hash_recursive(e, nested_hash)
      elsif e.element_children.length == 1
        # If there is one child, store the hash of the child
        nested_hash = {}
        hash[e.name.underscore.to_sym] = to_hash_recursive(e, nested_hash)
      else
        # If there are no children just get the content
        hash = add_content(e, hash)
      end
    end

    return hash.slice(*attribute_list(klass))
  end

  # Adds the pwml node content to the object hash
  # Removes any ids and replaces internal ids to other elements with the database ids
  # Also converts direction attributes
  # For all other attributes just stores the data as is
  def add_content(node, hash)
    if node.name != 'id' && node.content.present?
      if node.name == 'cell-type-id'
        hash[node.name.underscore.to_sym] = @cell_type_list[node.content]
      elsif node.name == 'species-id'
        hash[node.name.underscore.to_sym] = @species_list[node.content]
      elsif node.name == 'subcellular-location-id'
        hash[node.name.underscore.to_sym] = @subcellular_location_list[node.content]
      elsif node.name == 'tissue-id'
        hash[node.name.underscore.to_sym] = @tissue_list[node.content]
      elsif node.name == 'biological-state-id'
        hash[node.name.underscore.to_sym] = @biological_state_list[node.content]
      elsif node.name == 'edge-id'
        hash[node.name.underscore.to_sym] = @edge_list[node.content]
      elsif node.name == 'bound-id'
        hash[node.name.underscore.to_sym] = @bound_list[node.content]
      elsif node.name == 'bound-visualization-id'
        hash[node.name.underscore.to_sym] = @bound_visualization_list[node.content]
      elsif node.name == 'compound-id'
        hash[node.name.underscore.to_sym] = @compound_list[node.content]
      elsif node.name == 'compound-location-id'
        hash[node.name.underscore.to_sym] = @compound_location_list[node.content]
      elsif node.name == 'element-collection-id'
        hash[node.name.underscore.to_sym] = @element_collection_list[node.content]
      elsif node.name == 'element-collection-location-id'
        hash[node.name.underscore.to_sym] = @element_collection_location_list[node.content]
      elsif node.name == 'nucleic-acid-id'
        hash[node.name.underscore.to_sym] = @nucleic_acid_list[node.content]
      elsif node.name == 'nucleic-acid-location-id'
        hash[node.name.underscore.to_sym] = @nucleic_acid_location_list[node.content]
      elsif node.name == 'protein-id'
        hash[node.name.underscore.to_sym] = @protein_list[node.content]
      elsif node.name == 'protein-location-id'
        hash[node.name.underscore.to_sym] = @protein_location_list[node.content]
      elsif node.name == 'protein-complex-id'
        hash[node.name.underscore.to_sym] = @protein_complex_list[node.content]
      elsif node.name == 'protein-complex-visualization-id'
        hash[node.name.underscore.to_sym] = @protein_complex_visualization_list[node.content]
      elsif node.name == 'reaction-id'
        hash[node.name.underscore.to_sym] = @reaction_list[node.content]
      elsif node.name == 'reaction-coupled-transport-id'
        hash[node.name.underscore.to_sym] = @reaction_coupled_transport_list[node.content]
      elsif node.name == 'transport-id'
        hash[node.name.underscore.to_sym] = @transport_list[node.content]
      elsif node.name == 'interaction-id'
        hash[node.name.underscore.to_sym] = @interaction_list[node.content]
      elsif node.name == 'sub-pathway-id'
        hash[node.name.underscore.to_sym] = @sub_pathway_list[node.content]
      elsif node.name == 'direction'
        hash[node.name.underscore.to_sym] = convert_direction(node.content)
      else
        hash[node.name.underscore.to_sym] = node.content
      end
    end

    return hash
  end

  # To convert the direction for the database, since it should be in text in the pwml file
  def convert_direction(direction)
    if direction == "Right"
      return ">"
    elsif direction == "Left"
      return "<"
    elsif direction == "Both"
      return "<>"
    end
    return ""
  end

  # Get the list of valid attributes for that class
  # This is to remove any erroneous attributes from the pwml document so they don't cause errors on create
  def attribute_list(klass)
    klass.new.attributes.keys.map(&:to_sym)
  end

  # Stores the object in the provided hash
  # If it already exists in the database, get and store that object
  # Otherwise create a new object from the pwml data and store
  def store_object(node)
    if node.xpath('id').present?
      begin
        id = node.xpath('id').first.content
        hash = instance_variable_get("@#{node.name.underscore}_list")
        klass = node.name.underscore.camelize.constantize
        attribute_hash = to_hash_shallow(node).slice(*attribute_list(klass))

        if klass == SubPathway 
          # Don't check for existing subpathways since these will always be created new for the new pathway
          existing = nil
        else
          # Use the find_by method for simple objects
          existing = klass.find_by(attribute_hash)
        end
        if existing.present?
          hash[id] = existing.id
        else
          # Create a new object and store the id doesn't already exist
          new_object = klass.new(attribute_hash)
          if new_object.save
            hash[id] = new_object.id
          else
            @result[:errors]["#{node.name.underscore}_#{id.to_s}".to_sym] = new_object.errors.full_messages
          end
        end
        return
      rescue Exception => e
        # If there is an exception, it means the attribute value was invalid
        # So don't store anything, just return
        puts e.message
        puts e.backtrace.join("\n")
        return
      end
    end
  end

  # Stores the object in the provided hash, for nested objects so must be protein_complex, bound, reaction, reaction coupled transport, interaction, transport, sub_pathway
  # If it already exists in the database, get and store that object
  # Otherwise create a new object from the pwml data and store
  def store_nested_object(node, parent = nil)
    if node.xpath('id').present?
      begin
        id = node.xpath('id').first.content
        hash = instance_variable_get("@#{node.name.underscore}_list")
        klass = node.name.underscore.camelize.constantize
        attribute_hash = to_hash_shallow(node).slice(*attribute_list(klass))

        object = klass.new(attribute_hash)

        if parent.present?
          if object.has_attribute?("#{parent.class.to_s.underscore}_id".to_sym)
            object.send("#{parent.class.to_s.underscore}=", parent)
          elsif object.has_attribute?(:element_id)
            object.element = parent
          end
        end

        # This will get the nested attributes, only works for a single level
        node.element_children.each do |e|
          if e.element_children.length > 0
            e.element_children.each do |ec|
              child = get_child_object(ec, object)
              if child.present?
                object.send(e.name.underscore.to_sym) << child
              end
            end
          end
        end
        if klass == SubPathway # Don't check for existing subpathways since these will always be created new for the new pathway
          existing = nil
        else
          # Generate the signature and use the find_existing function for nested objects
          object.generate_signature
          existing = klass.find_existing(object)
        end
        if existing.present?
          hash[id] = existing.id
        elsif object.save
          # Save and store the new object id if it doesn't already exist
          hash[id] = object.id
        else
          @result[:errors]["#{node.name.underscore}_#{id.to_s}".to_sym] = object.errors.full_messages
        end
        return
      rescue Exception => e
        # If there is an exception, it means the attribute was invalid
        # So don't store anything, just return
        puts e.message
        puts e.backtrace.join("\n")
        return
      end
    end
  end

  # Stores the visualization in the provided hash, we don't need to check if visualizations exist
  def store_nested_visualization(node, pathway_visualization)
    if node.xpath('id').present?
      begin
        id = node.xpath('id').first.content
        hash = instance_variable_get("@#{node.name.underscore}_list")
        klass = node.name.underscore.camelize.constantize
        attribute_hash = to_hash_shallow(node).slice(*attribute_list(klass))
        
        object = klass.new(attribute_hash)

        # This will get the nested attributes, only works for a single level
        node.element_children.each do |e|
          if e.element_children.length > 0
            e.element_children.each do |ec|
              child = get_child_object(ec, object)
              if child.present?
                object.send(e.name.underscore.to_sym) << child
              end
            end
          end
        end

        # Set the pathway visualization, since all visualizations have this attribute
        object.pathway_visualization = pathway_visualization

        if object.save
          hash[id] = object.id
        else
          @result[:errors]["#{node.name.underscore}_#{id.to_s}".to_sym] = object.errors.full_messages
        end

        return
      rescue Exception => e
        # If there is an exception, it means the attribute was invalid
        # So don't store anything, just return
        puts e.message
        puts e.backtrace.join("\n")
        return
      end
    end
  end

  # Gets the child object but doesn't save it, since we are adding it to a parent object
  def get_child_object(node, parent)
    if node.xpath('id').present?
      begin
        id = node.xpath('id').first.content
        klass = node.name.underscore.camelize.constantize
        attribute_hash = to_hash_shallow(node).slice(*attribute_list(klass))

        object = klass.new(attribute_hash)

        # Set the parent association
        if object.has_attribute?("#{parent.class.to_s.underscore}_id".to_sym)
          object.send("#{parent.class.to_s.underscore}=", parent)
        elsif object.has_attribute?(:element_id)
          object.element = parent
        end

        if object.valid?
          return object
        else
          @result[:errors]["#{node.name.underscore}_#{id.to_s}".to_sym] = object.errors.full_messages
          return nil
        end
      rescue Exception => e
        # If there is an exception, it means the attribute value was invalid
        # So don't store anything, just return
        puts e.message
        puts e.backtrace.join("\n")
        return nil
      end
    end
  end

  # Builds visualization
  def build_nested_visualization(node, pathway_visualization)
    object = build_nested_visualization_recursive(node, pathway_visualization)
    return object
  end

  # Builds visualization but doesn't store or save it, similar to store_nested_visualization but recursive
  def build_nested_visualization_recursive(node, parent)
    begin
      id = node.xpath('id').first.try(:content)
      klass = node.name.underscore.camelize.constantize
      attribute_hash = to_hash_shallow(node).slice(*attribute_list(klass))
      object = klass.new(attribute_hash)

      # This will get the nested attributes only
      if node.element_children.present?
        node.element_children.each do |e|
          if e.element_children.length > 0
            e.element_children.each do |ec|
              child = build_nested_visualization_recursive(ec, object)
              if child.present?
                object.send(e.name.underscore.to_sym) << child
              end
            end
          end
        end
      end

      # Set the parent association
      if object.has_attribute?("#{parent.class.to_s.underscore}_id".to_sym)
        object.send("#{parent.class.to_s.underscore}=", parent)
      elsif object.has_attribute?(:element_id)
        object.element = parent
      end

      if object.valid?
        return object
      else
        @result[:errors]["#{node.name.underscore}_#{id.to_s}".to_sym] = object.errors.full_messages
        return nil
      end
    rescue Exception => e
      # If there is an exception, it means the attribute was invalid
      # So don't store anything, just return
      puts e.message
      puts e.backtrace.join("\n")
      return nil
    end
  end
   
end