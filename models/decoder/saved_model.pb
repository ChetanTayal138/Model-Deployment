��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02unknown8��
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
 
�
trainable_variables
	variables
&metrics

'layers
(layer_metrics
)non_trainable_variables
	regularization_losses
*layer_regularization_losses
 
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
+metrics

,layers
-layer_metrics
regularization_losses
.non_trainable_variables
/layer_regularization_losses
 
 
 
�
trainable_variables
	variables
0metrics

1layers
2layer_metrics
regularization_losses
3non_trainable_variables
4layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
trainable_variables
	variables
5metrics

6layers
7layer_metrics
regularization_losses
8non_trainable_variables
9layer_regularization_losses
 
 
 
�
trainable_variables
	variables
:metrics

;layers
<layer_metrics
regularization_losses
=non_trainable_variables
>layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
�
"trainable_variables
#	variables
?metrics

@layers
Alayer_metrics
$regularization_losses
Bnon_trainable_variables
Clayer_regularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_2Placeholder*/
_output_shapes
:��������� *
dtype0*$
shape:��������� 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias*
Tin
	2*
Tout
2*/
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_2396
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*&
f!R
__inference__traced_save_2557
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias*
Tin
	2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_restore_2587��
�
�
A__inference_model_2_layer_call_and_return_conditional_losses_2279
input_2
conv2d_2_2261
conv2d_2_2263
conv2d_3_2267
conv2d_3_2269
conv2d_4_2273
conv2d_4_2275
identity�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_2261conv2d_2_2263*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_21652"
 conv2d_2/StatefulPartitionedCall�
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_21882
up_sampling2d/PartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_2267conv2d_3_2269*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_22062"
 conv2d_3/StatefulPartitionedCall�
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_22292!
up_sampling2d_1/PartitionedCall�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_4_2273conv2d_4_2275*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_22472"
 conv2d_4/StatefulPartitionedCall�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
H
,__inference_up_sampling2d_layer_call_fn_2194

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_21882
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�"
�
 __inference__traced_restore_2587
file_prefix$
 assignvariableop_conv2d_2_kernel$
 assignvariableop_1_conv2d_2_bias&
"assignvariableop_2_conv2d_3_kernel$
 assignvariableop_3_conv2d_3_bias&
"assignvariableop_4_conv2d_4_kernel$
 assignvariableop_5_conv2d_4_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_2_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_2_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_4_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_4_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
J
.__inference_up_sampling2d_1_layer_call_fn_2235

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_22292
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
A__inference_model_2_layer_call_and_return_conditional_losses_2300
input_2
conv2d_2_2282
conv2d_2_2284
conv2d_3_2288
conv2d_3_2290
conv2d_4_2294
conv2d_4_2296
identity�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_2282conv2d_2_2284*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_21652"
 conv2d_2/StatefulPartitionedCall�
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_21882
up_sampling2d/PartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_2288conv2d_3_2290*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_22062"
 conv2d_3/StatefulPartitionedCall�
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_22292!
up_sampling2d_1/PartitionedCall�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_4_2294conv2d_4_2296*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_22472"
 conv2d_4/StatefulPartitionedCall�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
&__inference_model_2_layer_call_fn_2377
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_23622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�#
�
__inference__traced_save_2557
file_prefix.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2c7a3e9d1475423c9c85bad7d75a9242/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*_
_input_shapesN
L: :  : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
�	
�
&__inference_model_2_layer_call_fn_2512

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_23622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
|
'__inference_conv2d_4_layer_call_fn_2257

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_22472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
|
'__inference_conv2d_2_layer_call_fn_2175

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_21652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
A__inference_model_2_layer_call_and_return_conditional_losses_2362

inputs
conv2d_2_2344
conv2d_2_2346
conv2d_3_2350
conv2d_3_2352
conv2d_4_2356
conv2d_4_2358
identity�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_2344conv2d_2_2346*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_21652"
 conv2d_2/StatefulPartitionedCall�
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_21882
up_sampling2d/PartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_2350conv2d_3_2352*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_22062"
 conv2d_3/StatefulPartitionedCall�
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_22292!
up_sampling2d_1/PartitionedCall�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_4_2356conv2d_4_2358*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_22472"
 conv2d_4/StatefulPartitionedCall�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2165

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_2188

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(2
resize/ResizeNearestNeighbor�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�1
�
A__inference_model_2_layer_call_and_return_conditional_losses_2437

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource
identity��
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/Reluu
up_sampling2d/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape�
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack�
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1�
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2�
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const�
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul�
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:��������� *
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_3/Reluy
up_sampling2d_1/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape�
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack�
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1�
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2�
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const�
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul�
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:��������� *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d_4/BiasAdd�
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
conv2d_4/Sigmoidp
IdentityIdentityconv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� :::::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�1
�
A__inference_model_2_layer_call_and_return_conditional_losses_2478

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource
identity��
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_2/Reluu
up_sampling2d/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape�
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack�
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1�
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2�
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const�
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul�
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:��������� *
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_3/Reluy
up_sampling2d_1/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape�
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack�
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1�
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2�
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const�
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul�
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:��������� *
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d_4/BiasAdd�
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
conv2d_4/Sigmoidp
IdentityIdentityconv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� :::::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
A__inference_model_2_layer_call_and_return_conditional_losses_2324

inputs
conv2d_2_2306
conv2d_2_2308
conv2d_3_2312
conv2d_3_2314
conv2d_4_2318
conv2d_4_2320
identity�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_2306conv2d_2_2308*
Tin
2*
Tout
2*/
_output_shapes
:��������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_21652"
 conv2d_2/StatefulPartitionedCall�
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_21882
up_sampling2d/PartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_2312conv2d_3_2314*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_22062"
 conv2d_3/StatefulPartitionedCall�
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_22292!
up_sampling2d_1/PartitionedCall�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_4_2318conv2d_4_2320*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_22472"
 conv2d_4/StatefulPartitionedCall�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
&__inference_model_2_layer_call_fn_2339
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_23242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2206

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2229

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(2
resize/ResizeNearestNeighbor�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�7
�
__inference__wrapped_model_2153
input_23
/model_2_conv2d_2_conv2d_readvariableop_resource4
0model_2_conv2d_2_biasadd_readvariableop_resource3
/model_2_conv2d_3_conv2d_readvariableop_resource4
0model_2_conv2d_3_biasadd_readvariableop_resource3
/model_2_conv2d_4_conv2d_readvariableop_resource4
0model_2_conv2d_4_biasadd_readvariableop_resource
identity��
&model_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&model_2/conv2d_2/Conv2D/ReadVariableOp�
model_2/conv2d_2/Conv2DConv2Dinput_2.model_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
model_2/conv2d_2/Conv2D�
'model_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv2d_2/BiasAdd/ReadVariableOp�
model_2/conv2d_2/BiasAddBiasAdd model_2/conv2d_2/Conv2D:output:0/model_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
model_2/conv2d_2/BiasAdd�
model_2/conv2d_2/ReluRelu!model_2/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
model_2/conv2d_2/Relu�
model_2/up_sampling2d/ShapeShape#model_2/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
model_2/up_sampling2d/Shape�
)model_2/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)model_2/up_sampling2d/strided_slice/stack�
+model_2/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_2/up_sampling2d/strided_slice/stack_1�
+model_2/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_2/up_sampling2d/strided_slice/stack_2�
#model_2/up_sampling2d/strided_sliceStridedSlice$model_2/up_sampling2d/Shape:output:02model_2/up_sampling2d/strided_slice/stack:output:04model_2/up_sampling2d/strided_slice/stack_1:output:04model_2/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#model_2/up_sampling2d/strided_slice�
model_2/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_2/up_sampling2d/Const�
model_2/up_sampling2d/mulMul,model_2/up_sampling2d/strided_slice:output:0$model_2/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
model_2/up_sampling2d/mul�
2model_2/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#model_2/conv2d_2/Relu:activations:0model_2/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:��������� *
half_pixel_centers(24
2model_2/up_sampling2d/resize/ResizeNearestNeighbor�
&model_2/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&model_2/conv2d_3/Conv2D/ReadVariableOp�
model_2/conv2d_3/Conv2DConv2DCmodel_2/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0.model_2/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
model_2/conv2d_3/Conv2D�
'model_2/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv2d_3/BiasAdd/ReadVariableOp�
model_2/conv2d_3/BiasAddBiasAdd model_2/conv2d_3/Conv2D:output:0/model_2/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
model_2/conv2d_3/BiasAdd�
model_2/conv2d_3/ReluRelu!model_2/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
model_2/conv2d_3/Relu�
model_2/up_sampling2d_1/ShapeShape#model_2/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
model_2/up_sampling2d_1/Shape�
+model_2/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_2/up_sampling2d_1/strided_slice/stack�
-model_2/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/up_sampling2d_1/strided_slice/stack_1�
-model_2/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/up_sampling2d_1/strided_slice/stack_2�
%model_2/up_sampling2d_1/strided_sliceStridedSlice&model_2/up_sampling2d_1/Shape:output:04model_2/up_sampling2d_1/strided_slice/stack:output:06model_2/up_sampling2d_1/strided_slice/stack_1:output:06model_2/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_2/up_sampling2d_1/strided_slice�
model_2/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_2/up_sampling2d_1/Const�
model_2/up_sampling2d_1/mulMul.model_2/up_sampling2d_1/strided_slice:output:0&model_2/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
model_2/up_sampling2d_1/mul�
4model_2/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor#model_2/conv2d_3/Relu:activations:0model_2/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:��������� *
half_pixel_centers(26
4model_2/up_sampling2d_1/resize/ResizeNearestNeighbor�
&model_2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_2/conv2d_4/Conv2D/ReadVariableOp�
model_2/conv2d_4/Conv2DConv2DEmodel_2/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0.model_2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
model_2/conv2d_4/Conv2D�
'model_2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/conv2d_4/BiasAdd/ReadVariableOp�
model_2/conv2d_4/BiasAddBiasAdd model_2/conv2d_4/Conv2D:output:0/model_2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
model_2/conv2d_4/BiasAdd�
model_2/conv2d_4/SigmoidSigmoid!model_2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
model_2/conv2d_4/Sigmoidx
IdentityIdentitymodel_2/conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� :::::::X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
"__inference_signature_wrapper_2396
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*/
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__wrapped_model_21532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:��������� 
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
|
'__inference_conv2d_3_layer_call_fn_2216

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_22062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�	
�
&__inference_model_2_layer_call_fn_2495

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_23242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_28
serving_default_input_2:0��������� D
conv2d_48
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�6
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
D_default_save_signature
*E&call_and_return_all_conditional_losses
F__call__"�3
_tf_keras_model�3{"class_name": "Model", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_4", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_4", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 32]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 7, 7, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
�
trainable_variables
	variables
regularization_losses
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
�
trainable_variables
	variables
regularization_losses
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
*O&call_and_return_all_conditional_losses
P__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
&metrics

'layers
(layer_metrics
)non_trainable_variables
	regularization_losses
*layer_regularization_losses
F__call__
D_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Qserving_default"
signature_map
):'  2conv2d_2/kernel
: 2conv2d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
+metrics

,layers
-layer_metrics
regularization_losses
.non_trainable_variables
/layer_regularization_losses
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
0metrics

1layers
2layer_metrics
regularization_losses
3non_trainable_variables
4layer_regularization_losses
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_3/kernel
: 2conv2d_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
5metrics

6layers
7layer_metrics
regularization_losses
8non_trainable_variables
9layer_regularization_losses
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
:metrics

;layers
<layer_metrics
regularization_losses
=non_trainable_variables
>layer_regularization_losses
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_4/kernel
:2conv2d_4/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
"trainable_variables
#	variables
?metrics

@layers
Alayer_metrics
$regularization_losses
Bnon_trainable_variables
Clayer_regularization_losses
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference__wrapped_model_2153�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_2��������� 
�2�
A__inference_model_2_layer_call_and_return_conditional_losses_2279
A__inference_model_2_layer_call_and_return_conditional_losses_2437
A__inference_model_2_layer_call_and_return_conditional_losses_2478
A__inference_model_2_layer_call_and_return_conditional_losses_2300�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_model_2_layer_call_fn_2512
&__inference_model_2_layer_call_fn_2495
&__inference_model_2_layer_call_fn_2339
&__inference_model_2_layer_call_fn_2377�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
'__inference_conv2d_2_layer_call_fn_2175�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_2188�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
,__inference_up_sampling2d_layer_call_fn_2194�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2206�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
'__inference_conv2d_3_layer_call_fn_2216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2229�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_up_sampling2d_1_layer_call_fn_2235�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2247�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
'__inference_conv2d_4_layer_call_fn_2257�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
1B/
"__inference_signature_wrapper_2396input_2�
__inference__wrapped_model_2153 !8�5
.�+
)�&
input_2��������� 
� ";�8
6
conv2d_4*�'
conv2d_4����������
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2165�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
'__inference_conv2d_2_layer_call_fn_2175�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2206�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
'__inference_conv2d_3_layer_call_fn_2216�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2247� !I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
'__inference_conv2d_4_layer_call_fn_2257� !I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
A__inference_model_2_layer_call_and_return_conditional_losses_2279� !@�=
6�3
)�&
input_2��������� 
p

 
� "?�<
5�2
0+���������������������������
� �
A__inference_model_2_layer_call_and_return_conditional_losses_2300� !@�=
6�3
)�&
input_2��������� 
p 

 
� "?�<
5�2
0+���������������������������
� �
A__inference_model_2_layer_call_and_return_conditional_losses_2437x !?�<
5�2
(�%
inputs��������� 
p

 
� "-�*
#� 
0���������
� �
A__inference_model_2_layer_call_and_return_conditional_losses_2478x !?�<
5�2
(�%
inputs��������� 
p 

 
� "-�*
#� 
0���������
� �
&__inference_model_2_layer_call_fn_2339~ !@�=
6�3
)�&
input_2��������� 
p

 
� "2�/+����������������������������
&__inference_model_2_layer_call_fn_2377~ !@�=
6�3
)�&
input_2��������� 
p 

 
� "2�/+����������������������������
&__inference_model_2_layer_call_fn_2495} !?�<
5�2
(�%
inputs��������� 
p

 
� "2�/+����������������������������
&__inference_model_2_layer_call_fn_2512} !?�<
5�2
(�%
inputs��������� 
p 

 
� "2�/+����������������������������
"__inference_signature_wrapper_2396� !C�@
� 
9�6
4
input_2)�&
input_2��������� ";�8
6
conv2d_4*�'
conv2d_4����������
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_2229�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_up_sampling2d_1_layer_call_fn_2235�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_2188�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
,__inference_up_sampling2d_layer_call_fn_2194�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������