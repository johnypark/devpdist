


""" Modified code from tensorflow.org
and added custom codes. 
"""

import tensorflow as tf
import warnings


def remove_squeezable_dimensions(labels, predictions, expected_rank_diff=0, name=None):
  
  with tf.keras.backend.name_scope(name or 'remove_squeezable_dimensions'):
    if not isinstance(predictions, tf.RaggedTensor):
      predictions = tf.convert_to_tensor(predictions)
    if not isinstance(labels, tf.RaggedTensor):
      labels = tf.convert_to_tensor(labels)
    predictions_shape = predictions.shape
    predictions_rank = predictions_shape.ndims
    labels_shape = labels.shape
    labels_rank = labels_shape.ndims
    if (labels_rank is not None) and (predictions_rank is not None):
      # Use static rank.
      rank_diff = predictions_rank - labels_rank
      if (rank_diff == expected_rank_diff + 1 and
          predictions_shape.dims[-1].is_compatible_with(1)):
        predictions = tf.squeeze(predictions, [-1])
      elif (rank_diff == expected_rank_diff - 1 and
            labels_shape.dims[-1].is_compatible_with(1)):
        labels = tf.squeeze(labels, [-1])
      return labels, predictions

    # Use dynamic rank.
    rank_diff = tf.rank(predictions) - tf.rank(labels)
    if (predictions_rank is None) or (
        predictions_shape.dims[-1].is_compatible_with(1)):
      predictions = tf.cond(
          tf.equal(expected_rank_diff + 1, rank_diff),
          lambda: tf.squeeze(predictions, [-1]),
          lambda: predictions)
    if (labels_rank is None) or (
        labels_shape.dims[-1].is_compatible_with(1)):
      labels = tf.cond(
          tf.equal(expected_rank_diff - 1, rank_diff),
          lambda: tf.squeeze(labels, [-1]),
          lambda: labels)
    return labels, predictions


def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight = None):
  y_pred_shape = y_pred.shape
  y_pred_rank = y_pred_shape.ndims
  if y_true is not None:
    y_true_shape = y_true.shape
    y_true_rank = y_true_shape.ndims
    if (y_true_rank is not None) and (y_pred_rank is not None):
      # Use static rank for `y_true` and `y_pred`.
      if (y_pred_rank - y_true_rank != 1) or y_pred_shape[-1] == 1:
        y_true, y_pred = remove_squeezable_dimensions(
            y_true, y_pred)
    else:
      # Use dynamic rank.
      rank_diff = tf.rank(y_pred) - tf.rank(y_true)
      squeeze_dims = lambda: remove_squeezable_dimensions(  # pylint: disable=g-long-lambda
          y_true, y_pred)
      is_last_dim_1 = tf.equal(1, tf.shape(y_pred)[-1])
      maybe_squeeze_dims = lambda: tf.cond(  # pylint: disable=g-long-lambda
          is_last_dim_1, squeeze_dims, lambda: (y_true, y_pred))
      y_true, y_pred = tf.cond(
          tf.equal(1, rank_diff), maybe_squeeze_dims, squeeze_dims)

  if sample_weight is None:
    return y_pred, y_true

  weights_shape = sample_weight.shape
  weights_rank = weights_shape.ndims
  if weights_rank == 0:  # If weights is scalar, do nothing.
    return y_pred, y_true, sample_weight

  if (y_pred_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - y_pred_rank == 1:
      sample_weight = tf.squeeze(sample_weight, [-1])
    elif y_pred_rank - weights_rank == 1:
      sample_weight = tf.expand_dims(sample_weight, [-1])
    return y_pred, y_true, sample_weight

  # Use dynamic rank.
  weights_rank_tensor = tf.rank(sample_weight)
  rank_diff = weights_rank_tensor - tf.rank(y_pred)
  maybe_squeeze_weights = lambda: tf.squeeze(sample_weight, [-1])

  def _maybe_expand_weights():
    expand_weights = lambda: tf.expand_dims(sample_weight, [-1])
    return tf.cond(
        tf.equal(rank_diff, -1), expand_weights, lambda: sample_weight)

  def _maybe_adjust_weights():
    return tf.cond(
        tf.equal(rank_diff, 1), maybe_squeeze_weights,
        _maybe_expand_weights)
  # squeeze or expand last dim of `sample_weight` if its rank differs by 1
  # from the new rank of `y_pred`.

  sample_weight = tf.cond(
      tf.equal(weights_rank_tensor, 0), lambda: sample_weight,
      _maybe_adjust_weights)

  return y_pred, y_true, sample_weight




class LossFunctionWrapper(tf.keras.losses.Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None,
               **kwargs):
  
    super().__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    
    if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
      y_pred, y_true = squeeze_or_expand_dimensions(y_pred, y_true)

    ag_fn = tf.__internal__.autograph.tf_convert(self.fn, tf.__internal__.autograph.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in self._fn_kwargs.items():
      config[k] = tf.keras.backend.eval(v) # if tf.keras.utils.tf_utils.is_tensor_or_variable(v) else v
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _constant_to_tensor(x, dtype):
    return tf.constant(x, dtype=dtype)


def _safe_mean(losses, num_present):
    total_loss = tf.reduce_sum(losses)
    return tf.math.divide_no_nan(total_loss, num_present, name='value')


def _num_elements(losses):
    with tf.keras.backend.name_scope('num_elements') as scope:
        return tf.cast(tf.size(losses, name=scope), dtype=losses.dtype)


def reduce_weighted_loss(weighted_losses, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
    """Reduces the individual weighted loss measurements."""
    if reduction == tf.keras.losses.Reduction.NONE:
        loss = weighted_losses
    else:
        loss = tf.reduce_sum(weighted_losses)
        if reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            loss = _safe_mean(loss, _num_elements(weighted_losses))
    return loss


def compute_weighted_loss(losses,
                          sample_weight=None,
                          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                          name=None):
  
  tf.keras.losses.Reduction.validate(reduction)

  if reduction == tf.keras.losses.Reduction.AUTO:
    reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
  if sample_weight is None:
    sample_weight = 1.0
  with tf.keras.backend.name_scope(name or 'weighted_loss'):
    # Save the `reduction` argument for loss normalization when distributing
    # to multiple replicas. Used only for estimator + v1 optimizer flow.
    tf.compat.v1.get_default_graph()._last_loss_reduction = reduction  # pylint: disable=protected-access

    #if not isinstance(losses,
    #                  (keras_tensor.KerasTensor, tf.RaggedTensor)):
    #  losses = tf.convert_to_tensor(losses)

    #if not isinstance(sample_weight,
    #                  (keras_tensor.KerasTensor, tf.RaggedTensor)):
    #  sample_weight = tf.convert_to_tensor(sample_weight)

    # Convert any non float dtypes to floats, to avoid it loss any precision for
    # dtype like int or bool.
    if not losses.dtype.is_floating:
      input_dtype = losses.dtype
      losses = tf.cast(losses, 'float32')
      input_casted = True
    else:
      input_casted = False
    sample_weight = tf.cast(sample_weight, losses.dtype)
    # Update dimensions of `sample_weight` to match with `losses` if possible.
    losses, _, sample_weight = squeeze_or_expand_dimensions(  # pylint: disable=unbalanced-tuple-unpacking
        losses, None, sample_weight)
    weighted_losses = tf.multiply(losses, sample_weight)

    # Apply reduction function to the individual weighted losses.
    loss = reduce_weighted_loss(weighted_losses, reduction)
    if input_casted:
      # Convert the result back to the input type.
      loss = tf.cast(loss, input_dtype)
    return loss


def expand_weight_range(normalized_1d_tensor, weight_range):
    #N = len(normalized_1d_tensor)
    # #sum = tf.reduce_sum(out)
    out = normalized_1d_tensor *(weight_range[1] - weight_range[0]) + weight_range[0]
    return out



def get_genera2distance(DistanceFilename, data_type):	
	distance = dict()### 'Genus <=> Genus' => distance
	with open(DistanceFilename, mode = 'rt') as file:
		for k, line in enumerate(file):
			if k > 0:
				columns = line.rstrip().split('\t')
				distance[f"{columns[0]} <=> {columns[1]}"] = float(columns[2])
	genera2distance = tf.lookup.StaticHashTable( ### 'Genus <=> Genus' => distance
		tf.lookup.KeyValueTensorInitializer(
			tf.constant(list(distance.keys()), dtype = tf.string), 
			tf.constant(list(distance.values()), dtype = data_type)
			),
		default_value = -1
		)
	return genera2distance


def get_distance_from_two_classes(TrueClass, PredClass, LOOKUPTB):
    if TrueClass.dtype is not tf.string:
      TrueClass = tf.strings.as_string(TrueClass)
    if PredClass.dtype is not tf.string:
      PredClass = tf.strings.as_string(PredClass)
    return tf.math.maximum(
			LOOKUPTB.lookup(tf.strings.reduce_join(
				axis = -1,
				inputs = tf.stack([TrueClass, PredClass], axis = 1),
				separator = ' <=> '
			)),
			LOOKUPTB.lookup(tf.strings.reduce_join(
				axis = -1,
				inputs = tf.stack([PredClass, TrueClass], axis = 1),
				separator = ' <=> '
			))
      )

def get_class2genus(ls_class, ls_genus):
    class2genus = tf.lookup.StaticHashTable( ### taxonID => Genus
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(ls_class), dtype = tf.int32), 
            tf.constant(list(ls_genus), dtype = tf.string)
            ),
            default_value = 'NOT A GENUS'
        )

    return class2genus


def get_taxonID2genus_from_df(df_PATH, keycol, valcol):
    import pandas as pd
    df = pd.read_table(df_PATH)
    S_scif = sorted(set(df[keycol]))
    mapping_scif2int = dict(zip(S_scif, range(len(S_scif))))
    df_by_scif = df.groupby(keycol, as_index= False).first() 
    ls_class = [mapping_scif2int[ele] for ele in df_by_scif[keycol]]
    ls_genus = [ele for ele in df_by_scif[valcol]]
    return get_class2genus(ls_class, ls_genus)


def get_key_for_g2d(TrueGenu, PredGenu):
    out = tf.strings.reduce_join(
				axis = -1,
				inputs = tf.stack([TrueGenu, PredGenu], axis = 1),
				separator = ' <=> '
			  )
    return out


def backend_categorical_crossentropy(target, output, path_distance, path_metadata, weight_range, from_logits=False, axis=-1):
  
      target = tf.convert_to_tensor(target)
      output = tf.convert_to_tensor(output)
      target.shape.assert_is_compatible_with(output.shape)
      
      g2d = get_genera2distance(path_distance, data_type = tf.float32)
      l2g = get_taxonID2genus_from_df(path_metadata, keycol ="scientificName", valcol ="genus")

      target_labels = tf.argmax(target, axis = 1, output_type = tf.int32)
      output_labels = tf.argmax(output, axis = 1, output_type = tf.int32)
      pdist_weight =  get_distance_from_two_classes(
                      TrueClass = l2g[target_labels],
                      PredClass = l2g[output_labels],
                      LOOKUPTB = g2d)
      pdist_weight = expand_weight_range(pdist_weight, weight_range)

      if hasattr(output, '_keras_logits'):
        output = output._keras_logits  # pylint: disable=protected-access
        if from_logits:
          warnings.warn(
          '"`categorical_crossentropy` received `from_logits=True`, but '
          'the `output` argument was produced by a sigmoid or softmax '
          'activation and thus does not represent logits. Was this intended?"',
          stacklevel=2)
        from_logits = True

      if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(
          labels=target, logits=output, axis=axis) 

      if (not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable)) and
          output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
        assert len(output.op.inputs) == 1
        output = output.op.inputs[0]
        return tf.nn.softmax_cross_entropy_with_logits(
          labels=target, logits=output, axis=axis) 

      # scale preds so that the class probas of each sample sum to 1
      output = output / tf.reduce_sum(output, axis, True)
      # Compute cross entropy from probabilities.
      epsilon_ = _constant_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
      output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)
      losses = -tf.reduce_sum(target * tf.math.log(output) , axis)
      return compute_weighted_loss(
          losses, pdist_weight)


def pd_categorical_crossentropy(y_true,
                             y_pred,
                             path_distance,
                             path_metadata,
                             weight_range,
                             from_logits=False,
                             label_smoothing=0.,
                             axis=-1):
  
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

  def _smooth_labels():
    num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

  y_true = tf.__internal__.smart_cond.smart_cond(label_smoothing, _smooth_labels,
                                 lambda: y_true)

  return backend_categorical_crossentropy(
      y_true, y_pred, path_distance, path_metadata, weight_range, from_logits=from_logits, axis=axis)


class PatristicDistanceCrossentropy(LossFunctionWrapper):
  """ Patristic Distance Crossentropy : Cross entropy loss that sample weights are patristic distance between
   the predicted class and the actual class. 
   input format of map_*_to_* is tensorflow lookup table. 
   genus_to_distance takes genus1 <=> genus2 as input, which can be generated by function :get_key_for_g2d

  """
  def __init__(self,
               path_distance,
               path_metadata,
               weight_range:list,
               from_logits=False,
               label_smoothing:float = 0.,
               axis=-1,
               reduction=tf.keras.losses.Reduction.AUTO,
               name:str = 'PatristicDistanceCrossentropy'):
   
    super().__init__(
        pd_categorical_crossentropy,
        name=name,
        path_distance = path_distance,
        path_metadata = path_metadata,
        weight_range = weight_range,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis)